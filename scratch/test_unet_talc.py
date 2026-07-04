import os
import sys
import numpy as np
import cv2
import torch

sys.path.append(os.getcwd())

from utils import PreProcessingMethods

pre_pr = PreProcessingMethods()
test_image_path = r"c:\Users\pxtuhen\Documents\repos\GeoAIprivate\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\2550374-2 10х.JPG"

print("Checking talc.pth loading and prediction on smp.Unet...")
try:
    arr = np.fromfile(test_image_path, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"image not found at {test_image_path}")
        
    print(f"Loaded image shape: {image_bgr.shape}")
    
    # Сегментируем тальк с помощью UNet (smp.Unet с весами talc.pth)
    unet_mask = pre_pr.segment_talc_unet(image_bgr, model_path="models/talc.pth")
    
    talc_pixels = np.sum(unet_mask > 0)
    total_pixels = unet_mask.size
    talc_pct = (talc_pixels / total_pixels) * 100.0
    
    print("SUCCESS: smp.Unet loaded and predicted successfully!")
    print(f"Talc percentage detected by smp.Unet: {talc_pct:.2f}%")
except Exception as e:
    import traceback
    print("FAILED to run smp.Unet:")
    traceback.print_exc()
