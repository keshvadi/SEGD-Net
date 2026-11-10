"""H&E-compatible geometric and mild HSV augmentations."""

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class HistologyAugment:
    """Apply random flips, right-angle rotations, scale, and HSV perturbation."""

    def __init__(
        self,
        image_size: int = 224,
        scale: tuple[float, float] = (0.9, 1.1),
        hue: float = 0.04,
        saturation: float = 0.08,
    ) -> None:
        self.image_size = image_size
        self.scale = scale
        self.hue = hue
        self.saturation = saturation
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> "torch.Tensor":
        array = np.asarray(image.convert("RGB"))
        if random.random() < 0.5:
            array = cv2.flip(array, 1)
        if random.random() < 0.5:
            array = cv2.flip(array, 0)
        array = np.rot90(array, random.randrange(4)).copy()
        factor = random.uniform(*self.scale)
        height, width = array.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, factor)
        array = cv2.warpAffine(
            array, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101
        )
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-self.hue, self.hue) * 180) % 180
        hsv[..., 1] = np.clip(
            hsv[..., 1] * (1 + random.uniform(-self.saturation, self.saturation)),
            0,
            255,
        )
        array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return self.to_tensor(Image.fromarray(array))


class HistologyEvaluationTransform:
    """Deterministic resize and ImageNet normalization."""

    def __init__(self, image_size: int = 224) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __call__(self, image: Image.Image):
        return self.transform(image.convert("RGB"))
