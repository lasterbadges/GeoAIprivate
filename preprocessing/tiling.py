"""
Модуль для нарезки (тайлинга) гигапиксельных панорам на фрагменты 512x512
и последующей сборки масок обратно с блендингом границ.
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def slice_into_tiles(image: np.ndarray, tile_size=512, overlap=64):
    """
    Режет изображение на тайлы размера tile_size x tile_size с перекрытием overlap.
    Возвращает список тайлов и их координаты (y_min, y_max, x_min, x_max).
    """
    h, w = image.shape[:2]
    tiles = []
    coords = []
    
    stride = tile_size - overlap
    
    y_centers = range(0, h - tile_size + stride, stride)
    x_centers = range(0, w - tile_size + stride, stride)
    
    y_starts = list(y_centers)
    if len(y_starts) == 0 or y_starts[-1] + tile_size < h:
        y_starts.append(h - tile_size)
    x_starts = list(x_centers)
    if len(x_starts) == 0 or x_starts[-1] + tile_size < w:
        x_starts.append(w - tile_size)
        
    for y_orig in y_starts:
        for x_orig in x_starts:
            y = y_orig
            x = x_orig
            
            # Сдвигаем окно назад, если оно выходит за пределы изображения
            y_end = y + tile_size
            if y_end > h:
                y_end = h
                y = max(0, h - tile_size)
                
            x_end = x + tile_size
            if x_end > w:
                x_end = w
                x = max(0, w - tile_size)
                
            tile = image[y:y_end, x:x_end]
            tiles.append(tile)
            coords.append((y, y_end, x, x_end))
            
    return tiles, coords


def stitch_tiles(tiles, coords, target_shape, tile_size=512):
    """
    Сшивает маски тайлов обратно в единое панорамное изображение.
    Использует взвешенное сложение на краях, чтобы избежать резких швов.
    """
    h, w = target_shape[:2]
    
    weight_kernel = np.ones((tile_size, tile_size), dtype=np.float32)
    border = 32
    for i in range(border):
        val = i / border
        weight_kernel[i, :] *= val
        weight_kernel[tile_size - 1 - i, :] *= val
        weight_kernel[:, i] *= val
        weight_kernel[:, tile_size - 1 - i] *= val
        
    stitched_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    
    for tile, (y_min, y_max, x_min, x_max) in zip(tiles, coords):
        if len(tile.shape) == 3:
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            tile_gray = tile
            
        tile_float = tile_gray.astype(np.float32)
        
        stitched_sum[y_min:y_max, x_min:x_max] += tile_float * weight_kernel
        weight_sum[y_min:y_max, x_min:x_max] += weight_kernel
        
    weight_sum[weight_sum == 0] = 1e-6
    final_mask = (stitched_sum / weight_sum).astype(np.uint8)
    
    return final_mask
