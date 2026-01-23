import os
import PIL
import torchvision
import numpy as np
import torch
import random
import albumentations as A
import copy
import cv2
import pandas as pd
import glob

import matplotlib.pyplot as plt
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers.utils import load_image


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()

class CustomDatasetWithBG(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        placeholder_token="*",
        template="a photo of a {}",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token

        self.image_paths = []
        self.image_paths += [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if is_image(file_path) and not 'bg' in file_path]

        self.image_paths = sorted(self.image_paths)

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.template = template

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process(self, image):
        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):
        example = {}

        example["placeholder_token"] = self.placeholder_token
        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = [torch.tensor(placeholder_index)]

        example["input_ids"] = [self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]]

        image = Image.open(self.image_paths[i % self.num_images])

        mask_path = self.image_paths[i % self.num_images].replace('.jpg', '_bg.jpg')
        mask = load_image(mask_path)
        mask = np.array(mask)
        mask = np.where(mask > 0, 1, 0)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)
        object_tensor = image_np * mask
        example["pixel_values"] = self.process(image_np)


        ref_object_tensor = Image.fromarray(object_tensor.astype('uint8')).resize((224, 224), resample=self.interpolation)
        ref_image_tenser = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)
        example["pixel_values_obj"] = self.get_tensor_clip()(ref_object_tensor)
        example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tenser)

        ref_seg_tensor = Image.fromarray(mask.astype('uint8') * 255)
        ref_seg_tensor = self.get_tensor_clip(normalize=False)(ref_seg_tensor)
        example["pixel_values_seg"] = torch.nn.functional.interpolate(ref_seg_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)

        return example
