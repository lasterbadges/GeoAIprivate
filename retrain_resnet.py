"""
Дообучение ResNet-18 с агрессивной аугментацией x3.
Аугментация покрывает: разный цвет, яркость, ч/б симуляцию,
поворот, зеркала, blur, шум, перспективные искажения.
"""
import os
import glob
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF

sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Задача 3. Скажи мне, кто твой шлиф")
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ore_resnet18.pth")

FOLDERS = [
    (os.path.join(BASE_DIR, "Фото руд по сортам. ч1", "Рядовые руды"),         1),
    (os.path.join(BASE_DIR, "Фото руд по сортам. ч1", "Труднообогатимые руды"),0),
    (os.path.join(BASE_DIR, "Фото руд по сортам. ч2", "рядовые"),              1),
    (os.path.join(BASE_DIR, "Фото руд по сортам. ч2", "тонкие"),               0),
]

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
CROP = 256
AUG_MULT = 3   # каждый оригинал → 3 варианта в эпоху


def read_bgr(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------- аугментация для train ----------
class HeavyAug:
    """Тяжёлая аугментация: цвет, яркость, ч/б, blur, шум, геометрия."""

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]

        # случайный кроп 256x256
        y = random.randint(0, max(0, h - CROP))
        x = random.randint(0, max(0, w - CROP))
        patch = img_rgb[y:y+CROP, x:x+CROP]
        if patch.shape[0] < CROP or patch.shape[1] < CROP:
            patch = cv2.resize(patch, (CROP, CROP))

        # геометрия
        if random.random() < 0.5:
            patch = cv2.flip(patch, 1)
        if random.random() < 0.5:
            patch = cv2.flip(patch, 0)
        k = random.randint(0, 3)
        if k:
            patch = np.rot90(patch, k).copy()

        # симуляция ч/б снимка (20% вероятность)
        if random.random() < 0.20:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            patch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # яркость / контраст
        alpha = random.uniform(0.6, 1.5)   # контраст
        beta  = random.randint(-50, 50)     # яркость
        patch = np.clip(patch.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # color jitter (случайный сдвиг каналов)
        if random.random() < 0.5:
            shifts = np.random.randint(-30, 30, size=3)
            patch = np.clip(patch.astype(np.int32) + shifts, 0, 255).astype(np.uint8)

        # тёмные / пересвеченные снимки
        if random.random() < 0.15:
            gamma = random.uniform(0.3, 0.7)  # очень тёмный
            lut = np.array([min(255, int(((i/255.0)**gamma)*255)) for i in range(256)], np.uint8)
            patch = cv2.LUT(patch, lut)
        elif random.random() < 0.10:
            gamma = random.uniform(1.5, 2.5)  # пересвет
            lut = np.array([min(255, int(((i/255.0)**gamma)*255)) for i in range(256)], np.uint8)
            patch = cv2.LUT(patch, lut)

        # blur или шум
        r = random.random()
        if r < 0.25:
            ksize = random.choice([3, 5])
            patch = cv2.GaussianBlur(patch, (ksize, ksize), 0)
        elif r < 0.40:
            noise = np.random.normal(0, random.uniform(5, 25), patch.shape).astype(np.int32)
            patch = np.clip(patch.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        # CLAHE (как в реальном пайплайне)
        if random.random() < 0.3:
            lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 4.0), tileGridSize=(8, 8))
            l = clahe.apply(l)
            patch = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

        return patch


class OreDataset(Dataset):
    def __init__(self, paths, labels, is_train=True):
        self.labels   = labels
        self.is_train = is_train
        self.aug      = HeavyAug()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"  Загружаем {len(paths)} изображений в RAM...")
        self.images = []
        for p in paths:
            bgr = read_bgr(p)
            self.images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def __len__(self):
        # x3 для train (без отдельного сохранения файлов)
        return len(self.images) * (AUG_MULT * 40 if self.is_train else 1)

    def __getitem__(self, idx):
        img_idx = idx % len(self.images)
        img     = self.images[img_idx]
        label   = self.labels[img_idx]

        if self.is_train:
            patch = self.aug(img)
        else:
            # центральный кроп
            h, w = img.shape[:2]
            y = (h - CROP) // 2
            x = (w - CROP) // 2
            patch = img[y:y+CROP, x:x+CROP]

        return self.to_tensor(patch), torch.tensor(label, dtype=torch.long)


def main():
    # собираем пути
    data_by_class = {0: [], 1: []}
    for folder, label in FOLDERS:
        if not os.path.isdir(folder):
            print(f"  [!] Папка не найдена: {folder}")
            continue
        for f in glob.glob(os.path.join(folder, "*")):
            if os.path.splitext(f)[1].lower() in EXTS:
                data_by_class[label].append(f)

    for lbl in [0, 1]:
        random.shuffle(data_by_class[lbl])
        name = "Рядовая" if lbl == 1 else "Труднообогатимая"
        print(f"  {name}: {len(data_by_class[lbl])} оригиналов")

    # разбивка 80/20
    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []
    for lbl in [0, 1]:
        paths = data_by_class[lbl]
        split = int(len(paths) * 0.8)
        train_paths  += paths[:split];    train_labels  += [lbl] * split
        val_paths    += paths[split:];    val_labels    += [lbl] * (len(paths) - split)

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)}")

    # модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Устройство: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # загружаем текущие веса чтобы дообучить, а не с нуля
    if os.path.exists(MODEL_OUT):
        try:
            model.load_state_dict(torch.load(MODEL_OUT, map_location="cpu", weights_only=False))
            print("  Загружены существующие веса — дообучение.")
        except Exception as e:
            print(f"  Веса не подошли ({e}), обучаем с ImageNet-pretraining.")
    else:
        print("  Файл весов не найден — обучаем с ImageNet-pretraining.")

    model = model.to(device)

    train_ds = OreDataset(train_paths, train_labels, is_train=True)
    val_ds   = OreDataset(val_paths,   val_labels,   is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)

    # разные lr для backbone и head
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = list(model.fc.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 2e-5},
        {"params": head_params,     "lr": 1e-4},
    ], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # заранее готовим val-патчи (5 кропов на изображение)
    print("  Подготовка val-патчей...")
    val_batch = []
    for p, lbl in zip(val_paths, val_labels):
        bgr = read_bgr(p)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        cs = CROP
        crops = [
            img[0:cs,   0:cs],
            img[0:cs,   w-cs:w],
            img[h-cs:h, 0:cs],
            img[h-cs:h, w-cs:w],
            img[(h-cs)//2:(h-cs)//2+cs, (w-cs)//2:(w-cs)//2+cs],
        ]
        val_batch.append((crops, lbl))

    best_acc = 0.0
    num_epochs = 25

    print("\n--- Начало обучения ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_acc = correct / total * 100

        # валидация
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for crops, lbl in val_batch:
                tensors = torch.stack([val_transform(c) for c in crops]).to(device)
                probs   = torch.softmax(model(tensors), dim=1).cpu().numpy()
                pred    = np.argmax(np.mean(probs, axis=0))
                if pred == lbl:
                    val_correct += 1

        val_acc = val_correct / len(val_batch) * 100
        lr_now  = optimizer.param_groups[1]["lr"]
        print(f"  Epoch {epoch+1:02d}/{num_epochs} | loss={running_loss/total:.4f} "
              f"| train={train_acc:.1f}% | val={val_acc:.1f}% | lr={lr_now:.2e}")

        if val_acc >= best_acc:
            best_acc = val_acc
            # сохраняем через tmp-файл, чтобы не конфликтовать с открытым Streamlit
            tmp_path = MODEL_OUT + ".tmp"
            torch.save(model.state_dict(), tmp_path)
            if os.path.exists(MODEL_OUT):
                os.remove(MODEL_OUT)
            os.rename(tmp_path, MODEL_OUT)
            print(f"           ✓ Сохранён (val={val_acc:.1f}%)")

    print(f"\n=== Готово. Лучшая val accuracy: {best_acc:.2f}% ===")
    print(f"    Модель: {MODEL_OUT}")


if __name__ == "__main__":
    main()