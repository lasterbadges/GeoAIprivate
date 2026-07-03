import argparse
import cv2
import numpy as np
from pathlib import Path

def mouse_callback(event, x, y, flags, param):
    global filled_mask, img_display, image
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w = filled_mask.shape
        # Маска для floodFill должна быть на 2 пикселя больше с каждой стороны
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        
        # Заливаем белым цветом (255)
        cv2.floodFill(filled_mask, flood_mask, (x, y), 255)
        
        # Визуализируем: подсвечиваем залитые области зеленым
        img_display = image.copy()
        img_display[filled_mask == 255] = [0, 255, 0]
        print(f"-> Точка принята: X={x}, Y={y}. Если закрасилось всё, что нужно, нажмите 'S'.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Не удалось открыть файл: {args.image}")
        exit(1)

    # Выделяем синий цвет (как в основном скрипте)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([140, 255, 255]))
    
    # Берем небольшое ядро (5), чтобы линии не слипались с краями
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    line_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # filled_mask будет хранить результаты наших кликов
    filled_mask = line_mask.copy()
    img_display = image.copy()

    win_name = "Click INSIDE the ore. Press 'S' to Save, 'Q' to Cancel"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("\n=== ИНСТРУКЦИЯ ===")
    print("1. Кликните ЛКМ внутрь той половины объекта, которая заливалась.")
    print("2. Кликните ЛКМ внутрь второй половины (которая оставалась черной).")
    print("3. Программа сама обойдет микро-разрывы благодаря кликам.")
    print("4. Когда весь объект станет зеленым, нажмите 'S' (английскую) на клавиатуре для сохранения.\n")

    while True:
        cv2.imshow(win_name, img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') or key == ord('S'):
            # Извлекаем только то, что мы реально залили кликами (белые зоны)
            final_mask = np.zeros_like(line_mask)
            final_mask[filled_mask == 255] = 255
            
            # Убираем исходные тонкие контуры разметки, оставляя чистое тело объекта
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(final_mask)
            cv2.drawContours(clean_mask, contours, -1, 255, thickness=cv2.FILLED)

            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.output, clean_mask)
            print(f"[УСПЕХ] Идеальная маска сохранена в: {args.output}")
            break
        elif key == ord('q') or key == ord('Q') or key == 27:
            print("Выход отменен.")
            break

    cv2.destroyAllWindows()
