import threading

import cv2
import numpy as np
import torch
from clip.clipseg import CLIPDensePredT
from torchvision import transforms

from roop.config.types import Frame
from roop.processors.base import BaseProcessor


class Mask_Clip2Seg(BaseProcessor):
    plugin_options: dict = None
    model_clip = None

    processorname = "mask_clip2seg"
    type = "mask"
    supports_batch = False
    batch_size_limit = 1
    supports_parallel_single_batch = True

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_clip is None:
            self.model_clip = CLIPDensePredT(version="ViT-B/16", reduce_dim=64, complex_trans_conv=True)
            self.model_clip.eval()
            self.model_clip.load_state_dict(
                torch.load("models/CLIP/rd64-uni-refined.pth", map_location=torch.device("cpu")),
                strict=False,
            )
        if getattr(self, "transform", None) is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((256, 256)),
            ])
        if getattr(self, "_clip_dilate_kernel", None) is None:
            self._clip_dilate_kernel = np.ones((5, 5), np.float32)
        if getattr(self, "_run_lock", None) is None:
            self._run_lock = threading.Lock()

        self.device = torch.device(self.plugin_options["devicename"])
        self.model_clip.to(self.device)

    def Run(self, img1, keywords: str) -> Frame:
        if keywords is None or len(keywords) < 1 or img1 is None:
            return img1

        source_image_small = cv2.resize(img1, (256, 256))
        img_mask = np.full((source_image_small.shape[0], source_image_small.shape[1]), 0, dtype=np.float32)
        mask_border = 1
        mask_blur = 5
        clip_blur = 5

        img_mask = cv2.rectangle(
            img_mask,
            (mask_border, mask_border),
            (256 - mask_border - 1, 256 - mask_border - 1),
            (255, 255, 255),
            -1,
        )
        img_mask = cv2.GaussianBlur(img_mask, (mask_blur * 2 + 1, mask_blur * 2 + 1), 0)
        img_mask /= 255

        prompts = [prompt.strip() for prompt in keywords.split(",") if prompt.strip()]
        if not prompts:
            return img_mask

        img = self.transform(source_image_small).unsqueeze(0).to(self.device)
        with self._run_lock:
            with torch.inference_mode():
                preds = self.model_clip(img.repeat(len(prompts), 1, 1, 1), prompts)[0]
        clip_mask = torch.sigmoid(preds[0][0])
        for index in range(1, len(prompts)):
            clip_mask += torch.sigmoid(preds[index][0])

        clip_mask = np.clip(clip_mask.detach().float().cpu().numpy(), 0, 1)
        clip_mask[clip_mask > 0.5] = 1.0
        clip_mask[clip_mask <= 0.5] = 0.0
        clip_mask = cv2.dilate(clip_mask, self._clip_dilate_kernel, iterations=1)
        clip_mask = cv2.GaussianBlur(clip_mask, (clip_blur * 2 + 1, clip_blur * 2 + 1), 0)

        img_mask *= clip_mask
        img_mask[img_mask < 0.0] = 0.0
        return img_mask

    def Release(self):
        self.model_clip = None
        self.device = None
