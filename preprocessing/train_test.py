"""
Тестовый скрипт обучения модели сегментации талька.
Используется для проверки работоспособности ML-окружения на локальном ПК.
Обучает U-Net (ResNet18) на 2 эпохи и сохраняет тестовые веса.
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loguru import logger

# Добавляем корень проекта в путь поиска модулей
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.dataset import TalcSegmentationDataset

try:
    import segmentation_models_pytorch as smp
except ImportError:
    logger.error("Библиотека segmentation-models-pytorch не найдена. Установите её через pip install segmentation-models-pytorch")
    sys.exit(1)


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    images_dir = BASE_DIR / "data" / "train_segmentation" / "images"
    masks_dir = BASE_DIR / "data" / "train_segmentation" / "masks"
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    logger.info("=== Запуск тестового обучения сегментации талька ===")
    
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])
    
    try:
        dataset = TalcSegmentationDataset(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            transform=train_transform
        )
        if len(dataset) == 0:
            logger.error("Нет данных для обучения в data/train_segmentation/. Запустите сначала предобработку!")
            return
            
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")
    
    logger.info("Инициализация модели U-Net (ResNet18)...")
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model = model.to(device)
    
    criterion = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 2
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (images, masks) in enumerate(dataloader, 1):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Эпоха [{epoch}/{epochs}] - Средний Loss (Dice): {avg_loss:.4f}")
        
    output_model_path = models_dir / "test_unet.pth"
    torch.save(model.state_dict(), str(output_model_path))
    logger.info(f"✅ Тестовое обучение завершено! Веса сохранены в {output_model_path}")


if __name__ == "__main__":
    main()
