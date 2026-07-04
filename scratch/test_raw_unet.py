import os
import sys
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

sys.path.append(os.getcwd())

from utils import ResNetUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_image_path = r"c:\Users\pxtuhen\Documents\repos\GeoAIprivate\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\2550374-2 10х.JPG"

print("Testing raw smp.Unet prediction (without CLAHE/normalize_for_inference)...")
try:
    arr = np.fromfile(test_image_path, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    # Инициализируем модель напрямую (как в оригинале)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load("models/talc.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    from preprocessing.tiling import slice_into_tiles, stitch_tiles
    
    # Нарезка оригинального BGR без CLAHE!
    tiles, coords = slice_into_tiles(image_bgr, tile_size=512, overlap=0)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    
    pred_tiles = []
    with torch.no_grad():
        for tile in tiles:
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_resized = cv2.resize(tile_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(tile_resized.transpose(2, 0, 1)).float() / 255.0
            
            # Нормализуем по ImageNet
            tensor = (tensor.to(device) - mean) / std
            tensor = tensor.unsqueeze(0)
            
            prob = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
            pred_tile_resized = (prob > 0.5).astype(np.uint8) * 255
            pred_tile = cv2.resize(pred_tile_resized, (512, 512), interpolation=cv2.INTER_NEAREST)
            pred_tiles.append(pred_tile)
            
    # Сшиваем
    h, w = image_bgr.shape[:2]
    reconstructed_mask = stitch_tiles(pred_tiles, coords, (h, w))
    
    talc_pct = (np.sum(reconstructed_mask > 0) / reconstructed_mask.size) * 100.0
    print(f"SUCCESS: Talc percentage detected without CLAHE: {talc_pct:.2f}%")
except Exception as e:
    import traceback
    traceback.print_exc()
