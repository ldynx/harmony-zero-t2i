import os
import sys
import argparse
import random
import glob
import numpy as np
import torch

from PIL import Image
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.blip_diffusion import BlipDiffusionPipeline

from blip_diffusion.pipeline_blip_diffusion import BlipDiffusionPipelineMulti
from blip_diffusion.sa_swap import SelfAttentionSwapMaskAuto, register_attention_editor


def main(
    data_dir: str,
    data_class: str,
    prompt: str,
    output_dir: str,
    pretrained_t2i_path: str = "Salesforce/blipdiffusion",
    swap_start_step: int = 40,
    swap_end_step: int = 60,
    swap_start_layer: int = 10,
    swap_end_layer: int = 16,
    seed: int = None,
    guidance_scale: int = 7.5,
    num_inference_steps: int = 100,
    negative_prompt: str ="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Pipeline
    blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
        pretrained_t2i_path, torch_dtype=torch.float16
    ).to("cuda")
    blip_diffusion_pipe_multi = BlipDiffusionPipelineMulti.from_pretrained(
        pretrained_t2i_path, torch_dtype=torch.float16
    ).to("cuda")

    # Prepare inputs
    batch_size = 1
    num_channels = blip_diffusion_pipe_multi.unet.config.in_channels
    scale_down_factor = 2 ** (len(blip_diffusion_pipe_multi.unet.config.block_out_channels) - 1)
    height = 512 // scale_down_factor
    width = 512 // scale_down_factor
    shape = (batch_size, num_channels, height, width)

    image = load_image(data_dir)
    save_dir = f"./{output_dir}/{prompt.replace(' ', '_')}"

    # Seed generator
    i = random.randint(0, 8888) if seed is None else seed
    generator = torch.Generator().manual_seed(i)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=blip_diffusion_pipe.unet.dtype)

    # Run baseline!
    baseline = blip_diffusion_pipe(
        prompt,
        image,
        data_class,
        data_class,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
        generator=generator,
        latents=latents,
    ).images

    # Run ours!
    editor = SelfAttentionSwapMaskAuto(
        step_idx=range(swap_start_step, swap_end_step + 1),
        layer_idx=range(swap_start_layer, swap_end_layer + 1),
        thres=0.5
    )
    register_attention_editor(blip_diffusion_pipe_multi, editor)
    ctx_begin_pos = blip_diffusion_pipe_multi.config.ctx_begin_pos
    num_query_tokens = blip_diffusion_pipe_multi.qformer.config.num_query_tokens
    editor.ref_token_idx = ctx_begin_pos + num_query_tokens
    editor.cur_token_idx = ctx_begin_pos + num_query_tokens

    ours = blip_diffusion_pipe_multi(
        ["", prompt],
        image,
        data_class,
        data_class,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
        generator=generator,
        latents=latents.expand(2, -1, -1, -1),
        use_orthog=True,
        use_swap=True,
    ).images

    # Save outputs
    os.makedirs(f"{save_dir}/{i}", exist_ok=True)
    baseline[0].save(f"{save_dir}/{i}/blip_diffusion_baseline.png")
    ours[1].save(f"{save_dir}/{i}/blip_diffusion_ours.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./example/01.jpg')
    parser.add_argument("--data_class", type=str, default='cat')
    parser.add_argument("--prompt", type=str, default='jumping')
    parser.add_argument("--output_dir", type=str, default='./outputs')
    args = parser.parse_args()

    main(data_dir=args.data_dir, data_class=args.data_class, prompt=args.prompt, output_dir=args.output_dir)
