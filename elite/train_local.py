import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate.logging import get_logger
from diffusers import LMSDiscreteScheduler
from huggingface_hub import HfFolder, whoami

from tqdm.auto import tqdm


from typing import Optional
from elite.train_global import th2image


class MapperLocal(nn.Module):
    def __init__(self,
         input_dim: int,
         output_dim: int,
    ):
        super(MapperLocal, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:])
            hidden_states += (hidden_state.unsqueeze(0),)
        hidden_states = torch.cat(hidden_states, dim=0).mean(dim=0)
        return hidden_states

value_local_list = []

def inj_forward_crossattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None):

    context = encoder_hidden_states
    hidden_states_local = hidden_states.clone()
    if context is not None:
        context_tensor = context["CONTEXT_TENSOR"]
    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)

    if context is not None:
        key = self.to_k_global(context_tensor)
        value = self.to_v_global(context_tensor)
    else:
        key = self.to_k(context_tensor)
        value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)


    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)

    if context is not None and "LOCAL" in context:
        # Perform cross attention with the local context
        query_local = self.to_q(hidden_states_local)
        key_local = self.to_k_local(context["LOCAL"])
        value_local = self.to_v_local(context["LOCAL"])

        query_local = self.reshape_heads_to_batch_dim(query_local)
        key_local = self.reshape_heads_to_batch_dim(key_local)
        value_local = self.reshape_heads_to_batch_dim(value_local)

        attention_scores_local = torch.matmul(query_local, key_local.transpose(-1, -2))
        attention_scores_local = attention_scores_local * self.scale
        attention_probs_local = attention_scores_local.softmax(dim=-1)

        # To extract the attmap of learned [w]
        index_local = context["LOCAL_INDEX"]
        index_local = index_local.reshape(index_local.shape[0], 1).repeat((1, self.heads)).reshape(-1)
        attention_probs_clone = attention_probs.clone().permute((0, 2, 1))
        attention_probs_mask = attention_probs_clone[torch.arange(index_local.shape[0]), index_local]
        # Normalize the attention map
        attention_probs_mask = attention_probs_mask.unsqueeze(2) / attention_probs_mask.max()

        if "LAMBDA" in context:
            _lambda = context["LAMBDA"]
        else:
            _lambda = 1

        attention_probs_local = attention_probs_local * attention_probs_mask * _lambda
        hidden_states += torch.matmul(attention_probs_local, value_local)
        value_local_list.append(value_local)

    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states


def inj_forward_crossattention_editor(editor, place_in_unet):
    def inj_forward_crossattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None):

        context = encoder_hidden_states
        hidden_states_local = hidden_states.clone()
        if context is not None:
            context_tensor = context["CONTEXT_TENSOR"]
        else:
            context_tensor = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)

        if context is not None:
            key = self.to_k_global(context_tensor)
            value = self.to_v_global(context_tensor)
        else:
            key = self.to_k(context_tensor)
            value = self.to_v(context_tensor)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)


        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * self.scale

        attention_probs = attention_scores.softmax(dim=-1)

        # hidden_states = torch.matmul(attention_probs, value)
        hidden_states = editor(
            query, key, value, attention_scores, attention_probs, context is not None, place_in_unet,
            self.sliceable_head_dim, scale=self.scale)

        if context is not None and "LOCAL" in context:
            # Perform cross attention with the local context
            query_local = self.to_q(hidden_states_local)
            key_local = self.to_k_local(context["LOCAL"])
            value_local = self.to_v_local(context["LOCAL"])

            query_local = self.reshape_heads_to_batch_dim(query_local)
            key_local = self.reshape_heads_to_batch_dim(key_local)
            value_local = self.reshape_heads_to_batch_dim(value_local)

            attention_scores_local = torch.matmul(query_local, key_local.transpose(-1, -2))
            attention_scores_local = attention_scores_local * self.scale
            attention_probs_local = attention_scores_local.softmax(dim=-1)

            # To extract the attmap of learned [w]
            index_local = context["LOCAL_INDEX"]
            index_local = index_local.reshape(index_local.shape[0], 1).repeat((1, self.heads)).reshape(-1)
            attention_probs_clone = attention_probs.clone().permute((0, 2, 1))
            attention_probs_mask = attention_probs_clone[torch.arange(index_local.shape[0]), index_local]
            # Normalize the attention map
            attention_probs_mask = attention_probs_mask.unsqueeze(2) / attention_probs_mask.max()

            if "LAMBDA" in context:
                _lambda = context["LAMBDA"]
            else:
                _lambda = 1

            attention_probs_local = attention_probs_local * attention_probs_mask * _lambda
            hidden_states += torch.matmul(attention_probs_local, value_local)
            value_local_list.append(value_local)

        if hidden_states.shape[0] != batch_size:
            # If not swapping, reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states
    return inj_forward_crossattention

# ------------------------------------------------------------------------------

logger = get_logger(__name__)


def save_progress(mapper, accelerator, args, step=None):
    logger.info("Saving embeddings")

    state_dict = accelerator.unwrap_model(mapper).state_dict()

    if step is not None:
        torch.save(state_dict, os.path.join(args.output_dir, f"local_mapper_{str(step).zfill(6)}.pt"))
    else:
        torch.save(state_dict, os.path.join(args.output_dir, "local_mapper.pt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--global_mapper_path", type=str, default=None,
        help="If not none, the training will start from the given checkpoints."
    )
    parser.add_argument(
        "--local_mapper_path", type=str, default=None,
        help="If not none, the training will start from the given checkpoints."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def freeze_params(params):
    for param in params:
        param.requires_grad = False

def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


@torch.no_grad()
def validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, mapper_local, vae, device, guidance_scale, seed=None, llambda=1, num_steps=100):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    uncond_input = tokenizer(
        [''] * example["pixel_values"].shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder({'input_ids':uncond_input.input_ids.to(device)})[0]

    if seed is None:
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64)
        )
    else:
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64), generator=generator,
        )

    latents = latents.to(example["pixel_values_clip"])
    scheduler.set_timesteps(num_steps)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["index"]

    image = F.interpolate(example["pixel_values_clip"], (224, 224), mode='bilinear')
    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    inj_embedding = inj_embedding[:, 0:1, :]
    encoder_hidden_states = text_encoder({'input_ids': example["input_ids"],
                                          "inj_embedding": inj_embedding,
                                          "inj_index": placeholder_idx})[0]

    image_obj = F.interpolate(example["pixel_values_obj"], (224, 224), mode='bilinear')
    image_features_obj = image_encoder(image_obj, output_hidden_states=True)
    image_embeddings_obj = [image_features_obj[0], image_features_obj[2][4], image_features_obj[2][8],
                            image_features_obj[2][12], image_features_obj[2][16]]
    image_embeddings_obj = [emb.detach() for emb in image_embeddings_obj]

    inj_embedding_local = mapper_local(image_embeddings_obj)
    mask = F.interpolate(example["pixel_values_seg"], (16, 16), mode='nearest')
    mask = mask[:, 0].reshape(mask.shape[0], -1, 1)
    inj_embedding_local = inj_embedding_local * mask

    for t in tqdm(scheduler.timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
                "LOCAL": inj_embedding_local,
                "LOCAL_INDEX": placeholder_idx.detach(),
                "LAMBDA": llambda
            }
        ).sample
        value_local_list.clear()
        latent_model_input = scheduler.scale_model_input(latents, t)

        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embeddings,
            }
        ).sample
        value_local_list.clear()
        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]

    return ret_pil_images


@torch.no_grad()
def validation_ours(
        example, tokenizer, image_encoder, text_encoder, unet, mapper, mapper_local, vae, device, guidance_scale,
        seed=None, llambda=1, num_steps=100):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    uncond_input = tokenizer(
        [''] * example["pixel_values"].shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder({'input_ids':uncond_input.input_ids.to(device)})[0]

    if seed is None:
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64)
        )
    else:
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64), generator=generator,
        )

    latents = latents.to(example["pixel_values_clip"])
    scheduler.set_timesteps(num_steps)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["index"]

    image = F.interpolate(example["pixel_values_clip"], (224, 224), mode='bilinear')
    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    inj_embedding = inj_embedding[:, 0:1, :]
    encoder_hidden_states = text_encoder({'input_ids': example["input_ids"],
                                          "inj_embedding": inj_embedding,
                                          "inj_index": placeholder_idx})[0]

    ################ Orchestration ################
    emb_length = inj_embedding.shape[1]
    for bsz, idx in enumerate(placeholder_idx):
        visual_embeds = encoder_hidden_states[bsz, idx:idx + emb_length]
        text_prompt = example["text"].split(example["placeholder_token"][0])[1].strip()
        projections = []
        selected_text_embeddings = []
        for i, token in enumerate(text_prompt.split(" ")):
            if token == "a" or token == "an":
                continue
            selected_text_embeddings += [encoder_hidden_states[bsz, idx + emb_length + 1 + i]]
        selected_text_embeddings = torch.stack(selected_text_embeddings, 0)
        text_embeds_basis, _ = torch.linalg.qr(selected_text_embeddings.permute(1, 0).float())
        text_embeds_basis = text_embeds_basis.permute(1, 0).half()
        for i in range(text_embeds_basis.shape[0]):
            embed = text_embeds_basis[i, :]
            projection = torch.einsum('o, n i, i -> n o', embed, visual_embeds, embed)
            projections.append(projection / torch.norm(embed) ** 2)
        visual_embeds_orthogonal = visual_embeds - torch.mean(torch.cat(projections, 0), 0, keepdim=True)
        encoder_hidden_states[bsz, idx:idx + emb_length] = visual_embeds_orthogonal
    ###############################################

    image_obj = F.interpolate(example["pixel_values_obj"], (224, 224), mode='bilinear')
    image_features_obj = image_encoder(image_obj, output_hidden_states=True)
    image_embeddings_obj = [image_features_obj[0], image_features_obj[2][4], image_features_obj[2][8],
                            image_features_obj[2][12], image_features_obj[2][16]]
    image_embeddings_obj = [emb.detach() for emb in image_embeddings_obj]

    inj_embedding_local = mapper_local(image_embeddings_obj)
    mask = F.interpolate(example["pixel_values_seg"], (16, 16), mode='nearest')
    mask = mask[:, 0].reshape(mask.shape[0], -1, 1)
    inj_embedding_local = inj_embedding_local * mask

    ############### Self-Attn Swap ################
    inj_embedding_obj = mapper(image_embeddings_obj)
    inj_embedding_obj = inj_embedding_obj[:, 0:1, :]
    _encoder_hidden_states = text_encoder({'input_ids': uncond_input.input_ids.to(device),
                                          "inj_embedding": inj_embedding_obj,
                                          "inj_index": torch.tensor([2])})[0]
    encoder_hidden_states = torch.cat([_encoder_hidden_states, encoder_hidden_states], 0)
    latents = latents.expand(2, -1, -1, -1)
    inj_embedding_local = inj_embedding_local.expand(2, -1, -1)
    placeholder_idx = placeholder_idx.expand(2)
    uncond_embeddings = uncond_embeddings.expand(2, -1, -1)
    ###############################################

    for t in tqdm(scheduler.timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
                "LOCAL": inj_embedding_local,
                "LOCAL_INDEX": placeholder_idx.detach(),
                "LAMBDA": llambda
            }
        ).sample
        value_local_list.clear()
        latent_model_input = scheduler.scale_model_input(latents, t)

        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embeddings,
            }
        ).sample
        value_local_list.clear()
        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]

    return ret_pil_images
