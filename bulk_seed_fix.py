import argparse
import json
from pathlib import Path
import cv2
import numpy as np

# Глобальные переменные для работы с UI и историей
filled_mask = None
img_display = None
image_raw = None
line_mask = None
mask_history = []  # Список для хранения предыдущих состояний маски

def mouse_callback(event, x, y, flags, param):
    global filled_mask, img_display, image_raw, mask_history
    if event == cv2.EVENT_LBUTTONDOWN:
        # Сохраняем текущее состояние маски в историю ПЕРЕД новым кликом
        mask_history.append(filled_mask.copy())
        
        h, w = filled_mask.shape
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        
        # Заливаем белым цветом (255) от точки клика
        cv2.floodFill(filled_mask, flood_mask, (x, y), 255)
        
        # Обновляем экран
        img_display = image_raw.copy()
        img_display[filled_mask == 255] = [0, 255, 0]
        print(f"   [Клик] Заливка в X={x}, Y={y}. (Нажмите 'Z' для отмены, если промахнулись)")

def main():
    global filled_mask, img_display, image_raw, line_mask, mask_history

    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="dataset", help="Корневая папка датасета")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    fix_json_path = output_dir / "needs_manual_fix.json"

    if not fix_json_path.exists():
        print(f"Файл {fix_json_path} не найден!")
        return

    with open(fix_json_path, "r", encoding="utf-8") as f:
        bad_files_data = json.load(f)

    if not bad_files_data:
        print("Список файлов для ручной правки пуст!")
        return

    manifest_path = output_dir / "manifest.json"
    manifest_map = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            for item in manifest_data:
                source_id = Path(item["image"]).stem
                manifest_map[source_id] = item["split"]

    win_name = "Fixer: 'S' - Save, 'Z' - Undo, 'Q' - Exit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    total_files = len(bad_files_data)
    print(f"=== Найдено файлов для ручной заливки: {total_files} ===")
    print("Клавиши: 'S' — сохранить и далее, 'Z' — отменить последний клик, 'Q' — выйти.\n")

    for idx, item in enumerate(bad_files_data):
        img_path = Path(item["file"])
        file_id = img_path.stem
        
        split = manifest_map.get(file_id, "train")
        out_mask_path = output_dir / split / "masks" / f"{file_id}.png"

        print(f"[{idx + 1}/{total_files}] Обработка: {img_path.name}")

        image_raw = cv2.imread(str(img_path))
        if image_raw is None:
            continue

        hsv = cv2.cvtColor(image_raw, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([140, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        line_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        filled_mask = line_mask.copy()
        img_display = image_raw.copy()
        mask_history = []  # Очищаем историю для нового файла

        next_file = False
        while not next_file:
            cv2.imshow(win_name, img_display)
            key = cv2.waitKey(1) & 0xFF
            
            # Нажата 'Z' (английская) — ОТМЕНА
            if key == ord('z') or key == ord('Z'):
                if mask_history:
                    # Восстанавливаем маску из истории
                    filled_mask = mask_history.pop()
                    # Перерисовываем экран
                    img_display = image_raw.copy()
                    img_display[filled_mask == 255] = [0, 255, 0]
                    print("   [Отмена] Последний клик стерт!")
                else:
                    print("   [Инфо] Нечего отменять, вы в самом начале.")
            
            # Нажата 'S' — СОХРАНЕНИЕ
            elif key == ord('s') or key == ord('S'):
                final_mask = np.zeros_like(line_mask)
                final_mask[filled_mask == 255] = 255
                
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                clean_mask = np.zeros_like(final_mask)
                cv2.drawContours(clean_mask, contours, -1, 255, thickness=cv2.FILLED)

                out_mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_mask_path), clean_mask)
                print(f"   [Успех] Маска сохранена.")
                next_file = True
                
            # Нажата 'Q' или Esc — ВЫХОД
            elif key == ord('q') or key == ord('Q') or key == 27:
                print("\nРабота прервана.")
                cv2.destroyAllWindows()
                return

    print("\nВсе файлы обработаны!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
