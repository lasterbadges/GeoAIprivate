"""
PyTorch Dataset для загрузки нормализованных шлифов и масок талька.
Поддерживает аугментации через библиотеку albumentations.
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from loguru import logger


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    try:
        nparr = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(nparr, flags)
    except Exception as e:
        logger.error(f"Ошибка при чтении {path}: {e}")
        return None


class OreClassificationDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ["regular", "complex", "talc"]
        self.samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_folder = self.root_dir / class_name
            if class_folder.exists():
                for f in class_folder.glob("*"):
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((f, class_idx))
                        
        logger.info(f"Создан Dataset классификации: загружено {len(self.samples)} файлов")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = imread_unicode(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            
        return image, label


class TalcSegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.samples = []
        
        # Находим маски
        mask_files = list(self.masks_dir.glob("*_mask.png"))
        for mask_path in mask_files:
            orig_name = mask_path.name.replace("_mask.png", ".JPG")
            orig_path = self.images_dir / orig_name
            if not orig_path.exists():
                orig_name = mask_path.name.replace("_mask.png", ".jpg")
                orig_path = self.images_dir / orig_name
                
            if orig_path.exists():
                self.samples.append((orig_path, mask_path))
                
        logger.info(f"Создан Dataset сегментации: загружено {len(self.samples)} пар")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = imread_unicode(str(img_path))
        mask = imread_unicode(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise FileNotFoundError(f"Ошибка загрузки пары: {img_path.name}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
                
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            if mask.dtype == torch.uint8:
                mask = mask.float()
            
        return image, mask
