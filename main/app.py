"""
🔬 Веб-интерфейс «Скажи мне, кто твой шлиф» на Gradio.
Предоставляет геологам интуитивный дашборд для классификации руды,
проверки качества шлифа (дефекты/царапины) и экспорта отчетов.
"""

import os
import sys
from pathlib import Path
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Добавляем корень проекта в пути поиска, чтобы импортировать наши скрипты
sys.path.append(str(Path(__file__).resolve().parent.parent))


# ============================================================
# ЗАГЛУШКИ ДЛЯ ИНТЕГРАЦИИ (Пока Ваня обучает модели)
# ============================================================
def mock_classify_ore(image_np, has_defects=False, defect_details=""):
    h, w = image_np.shape[:2]
    
    # Задаем цвета маски: Зеленый = Обычные сульфиды, Красный = Тонкие сульфиды, Синий = Тальк
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Синие зоны (Тальк)
    cv2.circle(mask_rgb, (w//3, h//2), min(h, w)//5, (0, 0, 255), -1)
    # 2. Зеленые зоны (Обычные сульфиды)
    cv2.rectangle(mask_rgb, (w//2, h//3), (w*3//4, h*2//3), (0, 255, 0), -1)
    # 3. Красные зоны (Тонкие сульфиды)
    cv2.circle(mask_rgb, (w*2//3, h*3//4), min(h, w)//8, (255, 0, 0), -1)

    # Случайные метрики для правдоподобности
    talc_pct = round(float(np.random.uniform(2.0, 18.0)), 1)
    sulfide_pct = round(float(np.random.uniform(15.0, 45.0)), 1)
    normal_pct = round(float(np.random.uniform(30.0, 70.0)), 1)
    fine_pct = 100.0 - normal_pct
    
    if talc_pct > 10.0:
        ore_class = "оталькованная"
    elif normal_pct >= fine_pct:
        ore_class = "рядовая"
    else:
        ore_class = "труднообогатимая"
        
    confidence = round(float(np.random.uniform(0.85, 0.98)), 2)
    
    conclusion = f"### 🪨 Результаты анализа:\n"
    conclusion += f"- Тип руды: **{ore_class.upper()}**\n"
    conclusion += f"- Содержание талька: **{talc_pct}%** (порог оталькования — 10%)\n"
    conclusion += f"- Сульфиды общие: **{sulfide_pct}%**\n"
    conclusion += f"  - Обычные срастания: **{normal_pct:.1f}%**\n"
    conclusion += f"  - Тонкие срастания: **{fine_pct:.1f}%**\n"
    
    return {
        "class": ore_class,
        "confidence": confidence,
        "talc_percent": talc_pct,
        "sulfide_percent": sulfide_pct,
        "normal_pct": normal_pct,
        "fine_pct": fine_pct,
        "mask_overlay": mask_rgb,
        "conclusion": conclusion
    }


# ============================================================
# ОСНОВНАЯ ЛОГИКА ИНТЕРФЕЙСА
# ============================================================
def process_single_image(image_path, opacity):
    if not image_path:
        return None, None, "⚠️ Загрузите изображение шлифа", "Не определено"
        
    has_defects = False
    defect_details = ""
    
    # Читаем оригинал
    original = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    res = mock_classify_ore(original_rgb, has_defects, defect_details)
    
    # Накладываем цветную маску поверх оригинала
    alpha = opacity / 100.0
    overlay = (original_rgb * (1.0 - alpha) + res["mask_overlay"] * alpha).astype(np.uint8)
    
    # Формируем таблицу метрик
    metrics_data = {
        "Параметр": [
            "Рекомендуемый сорт руды",
            "Уверенность модели",
            "Доля талька (%)",
            "Общие сульфиды (%)",
            "Обычные срастания (%)",
            "Тонкие срастания (%)",
            "Качество шлифа"
        ],
        "Значение": [
            res["class"].upper(),
            f"{res['confidence']*100:.0f}%",
            f"{res['talc_percent']}%",
            f"{res['sulfide_percent']}%",
            f"{res['normal_pct']:.1f}%",
            f"{res['fine_pct']:.1f}%",
            "Не проверялось"
        ]
    }
    df = pd.DataFrame(metrics_data)
    
    return overlay, df, res["conclusion"], "Отключен"


def batch_process_files(files):
    if not files:
        return None
        
    results = []
    for f in files:
        img = cv2.imread(f.name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mock_classify_ore(img_rgb)
        
        results.append({
            "Файл": Path(f.name).name,
            "Качество шлифа": "Не проверялось",
            "Сорт руды": res["class"].upper(),
            "Тальк (%)": res["talc_percent"],
            "Сульфиды (%)": res["sulfide_percent"]
        })
        
    df = pd.DataFrame(results)
    csv_path = Path("data/batch_report.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(csv_path), index=False, encoding='utf-8-sig')
    
    return df, str(csv_path)


# ============================================================
# ДИЗАЙН И ВЕРСТКА GRADIO UI
# ============================================================
CSS_STYLE = """
.container { max-width: 1400px; margin: 0 auto; }
.header-text { text-align: center; margin-bottom: 20px; }
.card-warning { background-color: #fef3c7; border-left: 4px solid #d97706; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
.card-success { background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"),
    css=CSS_STYLE,
    title="🔬 Скажи мне, кто твой шлиф"
) as demo:
    
    with gr.Column(elem_classes="container"):

        gr.Markdown(
            """
            # 🔬 «Скажи мне, кто твой шлиф»
            ### Экспресс-анализ обогатимости руд по панорамным микрофотографиям шлифов
            ---
            """,
            elem_classes="header-text"
        )
        
        with gr.Tabs():
            # --- ВКЛАДКА 1: ОДИНОЧНЫЙ АНАЛИЗ ---
            with gr.TabItem("🔍 Анализ одного снимка"):
                with gr.Row():
                    # Левая колонка: управление
                    with gr.Column(scale=1):
                        input_img = gr.Image(
                            label="📷 Загрузите микрофотографию шлифа",
                            type="filepath",
                            height=350
                        )
                        opacity_slider = gr.Slider(
                            minimum=0, maximum=100, value=40, step=5,
                            label="Прозрачность маски (%)"
                        )
                        analyze_btn = gr.Button(
                            "🔬 Начать анализ", variant="primary", size="lg"
                        )
                        
                    # Правая колонка: результаты сегментации
                    with gr.Column(scale=1):
                        output_img = gr.Image(
                            label="🎨 Результат сегментации (Сульфиды / Тальк)",
                            height=350
                        )
                        quality_badge = gr.Textbox(
                            label="Статус качества стекла",
                            interactive=False
                        )
                        
                with gr.Row():
                    # Нижняя панель: метрики и заключение
                    with gr.Column(scale=1):
                        metrics_table = gr.Dataframe(
                            label="📊 Количественные показатели",
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        conclusion_md = gr.Markdown(
                            "### 📋 Заключение\n*Загрузите изображение и нажмите 'Начать анализ'*",
                        )
                        
                # Легенда цветов
                gr.Markdown(
                    """
                    ### 🎨 Справка по сегментации:
                    | Цвет | Описание | Значение для обогащения |
                    |---|---|---|
                    | 🟢 **Зелёный** | Крупные сульфиды (обычные срастания) | Легко обогатимая фаза |
                    | 🔴 **Красный** | Тонкие сульфиды (замещённые/мелкие) | Трудно обогатимая фаза |
                    | 🔵 **Синий** | Тальк (нерудная матрица) | Приводит к оталькованию (класс "оталькованная" при >10%) |
                    """
                )

            # --- ВКЛАДКА 2: ПАКЕТНАЯ ОБРАБОТКА ---
            with gr.TabItem("📦 Пакетная обработка серий"):
                gr.Markdown("Загрузите группу снимков для автоматического расчёта партии руды без участия пользователя.")
                
                with gr.Row():
                    batch_files = gr.File(
                        label="Загрузить серию изображений",
                        file_count="multiple",
                        file_types=["image"]
                    )
                
                batch_btn = gr.Button("⚡ Начать обработку серии", variant="primary")
                
                with gr.Row():
                    batch_results = gr.Dataframe(
                        label="Результаты экспресс-анализа серии",
                        interactive=False
                    )
                with gr.Row():
                    batch_download = gr.File(
                        label="📥 Скачать итоговый отчёт (CSV)"
                    )

            # --- ВКЛАДКА 3: ОПИСАНИЕ МЕТОДОЛОГИИ ---
            with gr.TabItem("ℹ️ Методология анализа"):
                gr.Markdown(
                    """
                    ## Алгоритм работы системы
                    
                    Система классификации руд объединяет методы глубокого машинного обучения и классического компьютерного зрения:
                    
                    1. **Нормализация яркости**: Алгоритм CLAHE локально выравнивает освещение, убирая тени от линз микроскопа.
                    2. **Детекция талька**: Нейросеть U-Net сегментирует тёмную рассеянную фазу. Если площадь талька составляет > 10% от всей площади шлифа, руда автоматически относится к **оталькованному сорту** (наиболее критичный параметр).
                    3. **Анализ срастаний сульфидов**: Метод порогового разделения Оцу выделяет рудные минералы, а морфологический анализ связных компонентов классифицирует тип срастаний на крупные (обычные) и мелкие/замещённые (тонкие).
                    4. **Принятие решений**:
                       - Тальк > 10% -> **Оталькованная**
                       - Тальк <= 10% и преобладают обычные срастания -> **Рядовая**
                       - Тальк <= 10% и преобладают тонкие срастания -> **Труднообогатимая**

                    """
                )

    # Задаем обработчик клика для одиночного анализа
    analyze_btn.click(
        fn=process_single_image,
        inputs=[input_img, opacity_slider],
        outputs=[output_img, metrics_table, conclusion_md, quality_badge]
    )
    
    # Задаем обработчик для пакетной обработки
    batch_btn.click(
        fn=batch_process_files,
        inputs=[batch_files],
        outputs=[batch_results, batch_download]
    )

if __name__ == "__main__":
    # Запускаем локально на порту 7865
    demo.launch(server_name="127.0.0.1", server_port=7865, share=False)
