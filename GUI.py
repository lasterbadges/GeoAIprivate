import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import time
from pathlib import Path
import torch

# Добавляем корень проекта в путь
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from utils import PreProcessingMethods

try:
    from streamlit_image_comparison import image_comparison
    HAS_IMAGE_COMPARISON = True
except ImportError:
    HAS_IMAGE_COMPARISON = False

# Настройка страницы
st.set_page_config(
    page_title="🔬 Скажи мне, кто твой шлиф",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализируем состояние сессии для кнопки "Выход"
if "logged_out" not in st.session_state:
    st.session_state.logged_out = False

# Инициализация процессора
pre_pr = PreProcessingMethods()

# Путь к U-Net весам
MODEL_PATH = "models/test_unet.pth"
RESNET_MODEL_PATH = "models/ore_resnet18.pth"
FEEDBACK_CSV = "data/geologist_feedback.csv"
ACTIVE_LEARNING_DIR = "data/active_learning"
HISTORY_CSV = "data/analysis_history.csv"

# CSS стили для красивого дашборда
CSS_STYLE = """
<style>
/* Подключаем шрифт Inter */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Применяем шрифт ко всему приложению */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0b0f19 !important;
    color: #cbd5e1 !important;
}

/* Оформление боковой панели */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Оформление вкладок */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: #0f172a;
    padding: 8px;
    border-radius: 12px;
    border-bottom: none;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    background-color: transparent !important;
    border-radius: 8px;
    padding: 8px 16px;
    color: #9ca3af !important;
    font-weight: 600;
    border: none !important;
    transition: all 0.2s ease-in-out;
}
.stTabs [aria-selected="true"] {
    background-color: #10b981 !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

/* Кастомные Glassmorphic карточки метрик */
.metrics-container {
    display: flex;
    gap: 16px;
    margin: 20px 0;
}
.metric-glass-card {
    flex: 1;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s ease;
}
.metric-glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(16, 185, 129, 0.3);
    box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.1);
}
.metric-label {
    font-size: 13px;
    color: #9ca3af;
    font-weight: 600;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
}
.metric-sublabel {
    font-size: 11px;
    color: #6b7280;
    margin-top: 4px;
}

/* Светящиеся неоновые баннеры результатов */
.ore-banner-card {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}
.ore-banner-title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 6px;
    color: rgba(255, 255, 255, 0.7);
}
.ore-banner-value {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}
.ore-banner-desc {
    font-size: 13.5px;
    color: rgba(255, 255, 255, 0.85);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Стилизация конкретных типов руд */
.ore-ordinary {
    background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
    border-color: #10b981;
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.25), inset 0 1px 0 rgba(255,255,255,0.1);
}
.ore-ordinary .ore-banner-value {
    color: #10b981;
}
.ore-difficult {
    background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
    border-color: #f59e0b;
    box-shadow: 0 0 25px rgba(245, 158, 11, 0.25), inset 0 1px 0 rgba(255,255,255,0.1);
}
.ore-difficult .ore-banner-value {
    color: #f59e0b;
}
.ore-talc {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border-color: #6366f1;
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.25), inset 0 1px 0 rgba(255,255,255,0.1);
}
.ore-talc .ore-banner-value {
    color: #6366f1;
}

/* Стилизация легенды */
.legend-box {
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-weight: 600;
    color: white;
}

/* Стилизация кнопок Streamlit */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
}

/* Стилизация экспандеров */
.conda-container {
    background-color: #0f172a !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- ЭКРАН ВЫХОДА ---
if st.session_state.logged_out:
    st.warning("🔒 Сессия завершена. Вы успешно вышли из системы.")
    st.info("Вы можете безопасно закрыть эту вкладку браузера.")
    if st.button("Войти снова"):
        st.session_state.logged_out = False
        st.rerun()
    st.stop()


# --- ОСНОВНАЯ ЛОГИКА АНАЛИЗА ---
def analyze_single_image(img_path, brightness_thresh, talc_thresh, use_unet, use_resnet=True):
    # 1. Читаем и предобрабатываем изображение
    image_bgr, image_enhanced_gray, image_hsv = pre_pr.preprocess_image(img_path)
    
    # Конвертируем оригинал в RGB для отрисовки
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Создаем бинарную маску (проверяем синие линии)
    filled_mask = pre_pr.create_binary_mask(image_hsv)
    
    # 3. Запускаем классификацию сегментов
    # Передаем параметры
    final_mask = pre_pr.classify_ore_segments(
        filled_mask=filled_mask,
        image_enhanced_gray=image_enhanced_gray,
        brightness_threshold=brightness_thresh,
        talc_threshold=talc_thresh,
        image_bgr=image_bgr,
        use_unet=use_unet,
        model_path=MODEL_PATH,
        use_resnet=use_resnet,
        resnet_model_path=RESNET_MODEL_PATH
    )
    
    # 4. Расчет количественных метрик
    total_pixels = final_mask.size
    ordinary_pixels = np.sum(final_mask == 1)
    fine_pixels = np.sum(final_mask == 2)
    talc_pixels = np.sum(final_mask == 3)
    sulfide_pixels = ordinary_pixels + fine_pixels
    
    talc_pct = round(float(talc_pixels / total_pixels) * 100.0, 2)
    sulfide_pct = round(float(sulfide_pixels / total_pixels) * 100.0, 2)
    
    if sulfide_pixels > 0:
        ordinary_pct = round(float(ordinary_pixels / sulfide_pixels) * 100.0, 1)
        fine_pct = round(float(fine_pixels / sulfide_pixels) * 100.0, 1)
    else:
        ordinary_pct = 0.0
        fine_pct = 0.0
        
    # Экспертное правило классификации
    if talc_pct > 10.0:
        ore_class = "Оталькованная"
    elif ordinary_pixels >= fine_pixels:
        ore_class = "Рядовая"
    else:
        ore_class = "Труднообогатимая"
        
    # Создаем RGB маску для наложения
    # Зеленый = Обычные сульфиды (1), Красный = Тонкие сульфиды (2), Синий = Тальк (3)
    h, w = final_mask.shape[:2]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[final_mask == 1] = [0, 255, 0]   # Зеленый
    mask_rgb[final_mask == 2] = [255, 0, 0]   # Красный
    mask_rgb[final_mask == 3] = [0, 0, 255]   # Синий
    
    return {
        "class": ore_class,
        "talc_pct": talc_pct,
        "sulfide_pct": sulfide_pct,
        "ordinary_pct": ordinary_pct,
        "fine_pct": fine_pct,
        "original_rgb": original_rgb,
        "mask_rgb": mask_rgb,
        "final_mask": final_mask
    }


# --- СОХРАНЕНИЕ ОТЗЫВОВ ДЛЯ ACTIVE LEARNING ---
def save_geologist_feedback(filename, predicted_class, corrected_class, talc_pct, sulfide_pct, temp_img_path):
    # Записываем в CSV
    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    df = pd.DataFrame([{
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "PredictedClass": predicted_class,
        "CorrectedClass": corrected_class,
        "TalcPercent": talc_pct,
        "SulfidePercent": sulfide_pct
    }])
    
    if os.path.exists(FEEDBACK_CSV):
        df.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(FEEDBACK_CSV, index=False, encoding='utf-8-sig')
        
    # Копируем файл в папку активного обучения
    dest_folder = Path(ACTIVE_LEARNING_DIR) / corrected_class.lower()
    dest_folder.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(temp_img_path, dest_folder / filename)


# --- ЛОГИРОВАНИЕ РЕЗУЛЬТАТОВ АНАЛИЗА ---
def save_analysis_to_history(filename, res):
    os.makedirs(os.path.dirname(HISTORY_CSV), exist_ok=True)
    df = pd.DataFrame([{
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Class": res["class"],
        "TalcPercent": res["talc_pct"],
        "SulfidePercent": res["sulfide_pct"],
        "OrdinaryPercent": res["ordinary_pct"],
        "FinePercent": res["fine_pct"]
    }])
    
    if os.path.exists(HISTORY_CSV):
        try:
            df.to_csv(HISTORY_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
        except Exception:
            pass
    else:
        df.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')


# --- ИНТЕРФЕЙС STREAMLIT ---
# Шапка
col_title, col_exit = st.columns([6, 1])
with col_title:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 36px;
            font-weight: 800;
            margin: 0;
            padding-bottom: 5px;
        ">🔬 GeoAI: Скажи мне, кто твой шлиф</h1>
        <p style="
            color: #9ca3af;
            font-size: 15px;
            margin: 0;
            font-weight: 500;
        ">Интеллектуальный экспресс-анализ обогатимости медных и никелевых руд по микроструктуре шлифов</p>
    </div>
    """, unsafe_allow_html=True)
with col_exit:
    if st.button("🚪 Выход", use_container_width=True):
        st.session_state.logged_out = True
        st.rerun()

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Анализ одного снимка", 
    "📦 Пакетный анализ серий", 
    "📊 Аналитический дашборд",
    "ℹ️ Методология и Справка"
])

# --- ВКЛАДКА 1: ОДИНОЧНЫЙ АНАЛИЗ ---
with tab1:
    col_input, col_output = st.columns([2, 3])
    
    with col_input:
        st.subheader("📁 Входные данные")
        uploaded_file = st.file_uploader(
            "🖼️ Загрузите микрофотографию шлифа (PNG, JPG, JPEG)", 
            type=["png", "jpg", "jpeg"]
        )
        
        # Автоматические параметры ИИ-анализа
        use_resnet = True
        brightness_thresh = 120
        talc_thresh = 30
        use_unet = True
            
        opacity_slider = st.slider(
            "Прозрачность маски наложения (%)", 
            min_value=0, max_value=100, value=40, step=5
        )
        
        run_analysis = st.button("🚀 Начать анализ", type="primary", use_container_width=True)
        
    with col_output:
        st.subheader("📊 Результаты анализа")
        
        if run_analysis and uploaded_file is not None:
            # Сохраняем временно файл на диск для обработки
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Запускаем обработку с индикатором
            with st.spinner("Анализируем микроструктуру..."):
                start_time = time.time()
                res = analyze_single_image(
                    str(temp_path), 
                    brightness_thresh, 
                    talc_thresh, 
                    use_unet,
                    use_resnet
                )
                duration = time.time() - start_time
                save_analysis_to_history(uploaded_file.name, res)
                
            # Разный цвет баннера в зависимости от класса руды
            if res["class"] == "Оталькованная":
                st.markdown(f"""
                <div class="ore-banner-card ore-talc">
                    <div class="ore-banner-title">Рекомендуемый сорт руды</div>
                    <div class="ore-banner-value">ОТАЛЬКОВАННАЯ РУДА</div>
                    <div class="ore-banner-desc">Содержание талька превышает критический порог в 10% (содержание талька — {res['talc_pct']}%). Требуется введение депрессоров талька в цикле флотации.</div>
                </div>
                """, unsafe_allow_html=True)
            elif res["class"] == "Труднообогатимая":
                st.markdown(f"""
                <div class="ore-banner-card ore-difficult">
                    <div class="ore-banner-title">Рекомендуемый сорт руды</div>
                    <div class="ore-banner-value">ТРУДНООБОГАТИМАЯ РУДА</div>
                    <div class="ore-banner-desc">Доля талька в пределах нормы (содержание талька — {res['talc_pct']}%). Преобладают тонкие срастания сульфидов ({res['fine_pct']}%). Требуется более тонкое измельчение руды.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ore-banner-card ore-ordinary">
                    <div class="ore-banner-title">Рекомендуемый сорт руды</div>
                    <div class="ore-banner-value">РЯДОВАЯ РУДА</div>
                    <div class="ore-banner-desc">Доля талька в норме (содержание талька — {res['talc_pct']}%). Преобладают крупные сплошные сульфиды ({res['ordinary_pct']}%). Стандартные параметры обогащения.</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Отображаем изображение
            if HAS_IMAGE_COMPARISON:
                # Создаем полупрозрачную маску поверх оригинала
                alpha = opacity_slider / 100.0
                overlay = (res["original_rgb"] * (1.0 - alpha) + res["mask_rgb"] * alpha).astype(np.uint8)
                
                # Преобразуем в PIL.Image для компонента
                img_orig_pil = Image.fromarray(res["original_rgb"])
                img_overlay_pil = Image.fromarray(overlay)
                
                image_comparison(
                    img1=img_orig_pil,
                    img2=img_overlay_pil,
                    label1="Шлиф (Оригинал)",
                    label2="Анализ (Маска)",
                    show_labels=True,
                    make_responsive=True,
                    starting_position=50
                )
                st.caption(f"Сдвигайте центральный бегунок для сравнения. Время анализа: {duration:.2f} сек.")
            else:
                alpha = opacity_slider / 100.0
                overlay = (res["original_rgb"] * (1.0 - alpha) + res["mask_rgb"] * alpha).astype(np.uint8)
                st.image(
                    overlay, 
                    caption=f"Маска наложения ({uploaded_file.name}). Время анализа: {duration:.2f} сек.", 
                    use_container_width=True
                )
            
            # Метрики
            st.markdown(f"""
            <div class="metrics-container">
                <div class="metric-glass-card">
                    <div class="metric-label">Доля талька</div>
                    <div class="metric-value">{res['talc_pct']}%</div>
                    <div class="metric-sublabel">Порог: 10.0%</div>
                </div>
                <div class="metric-glass-card">
                    <div class="metric-label">Общие сульфиды</div>
                    <div class="metric-value">{res['sulfide_pct']}%</div>
                    <div class="metric-sublabel">От площади шлифа</div>
                </div>
                <div class="metric-glass-card">
                    <div class="metric-label">Обычные / Тонкие</div>
                    <div class="metric-value">{res['ordinary_pct']}% / {res['fine_pct']}%</div>
                    <div class="metric-sublabel">От суммы сульфидов</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
            # Сводная таблица показателей
            df_metrics = pd.DataFrame({
                "Параметр": [
                    "Сорт руды",
                    "Доля талька",
                    "Общая доля сульфидов",
                    "Обычные срастания (от сульфидов)",
                    "Тонкие срастания (от сульфидов)"
                ],
                "Значение": [
                    res["class"].upper(),
                    f"{res['talc_pct']}%",
                    f"{res['sulfide_pct']}%",
                    f"{res['ordinary_pct']}%",
                    f"{res['fine_pct']}%"
                ]
            })
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            # Ссылка на скачивание отчета
            report_text = f"Отчет анализа шлифа {uploaded_file.name}\n"
            report_text += f"Рекомендуемый сорт руды: {res['class'].upper()}\n"
            report_text += f"Доля талька: {res['talc_pct']}%\n"
            report_text += f"Доля сульфидов: {res['sulfide_pct']}%\n"
            report_text += f"Преобладание обычных срастаний: {res['ordinary_pct']}%\n"
            report_text += f"Преобладание тонких срастаний: {res['fine_pct']}%\n"
            
            st.download_button(
                "📥 Скачать текстовый отчет",
                data=report_text,
                file_name=f"Report_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )
            
            # --- РЕЖИМ ЭКСПЕРТНОЙ ПРОВЕРКИ (ACTIVE LEARNING) ---
            st.markdown("---")
            st.subheader("✍️ Экспертная верификация (Active Learning)")
            st.markdown("Если вы не согласны с решением алгоритма, укажите верный сорт руды. Файл автоматически запишется в обучающую выборку.")
            
            corrected_class = st.selectbox(
                "Фактический сорт руды (экспертная оценка):",
                ["Рядовая", "Труднообогатимая", "Оталькованная"],
                index=["Рядовая", "Труднообогатимая", "Оталькованная"].index(res["class"])
            )
            
            if st.button("💾 Сохранить оценку и отправить в обучающий пул"):
                save_geologist_feedback(
                    uploaded_file.name,
                    res["class"],
                    corrected_class,
                    res["talc_pct"],
                    res["sulfide_pct"],
                    str(temp_path)
                )
                st.success("✅ Отзыв успешно сохранен! Данные отправлены в базу дообучения моделей.")
                
        elif run_analysis:
            st.error("⚠️ Сначала выберите изображение шлифа для анализа.")
        else:
            st.info("💡 Загрузите изображение в левой колонке и нажмите 'Начать анализ'.")


# --- ВКЛАДКА 2: ПАКЕТНЫЙ АНАЛИЗ ---
with tab2:
    st.subheader("📦 Обработка партий шлифов")
    st.write("Загрузите несколько файлов для массового экспресс-расчета.")
    
    batch_files = st.file_uploader(
        "Выберите изображения для пакетной обработки:",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    # Автоматические параметры ИИ-анализа серии
    b_use_resnet = True
    b_brightness_thresh = 120
    b_talc_thresh = 30
    b_use_unet = True
        
    start_batch = st.button("⚡ Начать обработку серии", type="primary")
    
    if start_batch and batch_files:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Временная директория
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, file in enumerate(batch_files):
            status_text.text(f"Обработка файла {idx+1}/{len(batch_files)}: {file.name}")
            
            # Сохраняем временный файл
            temp_path = temp_dir / file.name
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
                
            # Запускаем расчет
            res = analyze_single_image(
                str(temp_path), 
                b_brightness_thresh, 
                b_talc_thresh, 
                b_use_unet,
                b_use_resnet
            )
            save_analysis_to_history(file.name, res)
            
            results.append({
                "Имя файла": file.name,
                "Рекомендуемый сорт": res["class"].upper(),
                "Тальк (%)": res["talc_pct"],
                "Сульфиды (%)": res["sulfide_pct"],
                "Обычные срастания (%)": res["ordinary_pct"],
                "Тонкие срастания (%)": res["fine_pct"]
            })
            
            # Обновляем прогресс
            progress_bar.progress((idx + 1) / len(batch_files))
            
        status_text.text("✅ Обработка серии завершена!")
        
        # Отображаем таблицу результатов
        df_batch = pd.DataFrame(results)
        st.dataframe(df_batch, use_container_width=True, hide_index=True)
        
        # Экспорт в CSV
        csv_data = df_batch.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Скачать сводный отчет (CSV)",
            data=csv_data,
            file_name="batch_report_ore_classification.csv",
            mime="text/csv"
        )
        
    elif start_batch:
        st.error("⚠️ Загрузите хотя бы один файл для пакетного анализа.")


# --- ВКЛАДКА 3: МЕТОДОЛОГИЯ ---
with tab3:
    st.subheader("ℹ️ Методология автоматической классификации шлифов")
    st.markdown("""
    Система объединяет алгоритмы **компьютерного зрения (OpenCV)** и **глубокого обучения (PyTorch U-Net)** для обеспечения максимальной геологической точности:
    
    ### 🎨 Легенда маски наложения:
    """)
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown('<div class="legend-box" style="background-color: rgba(0, 255, 0, 0.2); border-left: 5px solid #00ff00; color: #047857;">🟢 Обычные срастания (сульфиды)</div>', unsafe_allow_html=True)
        st.write("Крупные, изолированные сульфидные зерна с минимальным замещением силикатной/нерудной фазой. Маркер **рядовой руды**.")
    with col_l2:
        st.markdown('<div class="legend-box" style="background-color: rgba(255, 0, 0, 0.2); border-left: 5px solid #ff0000; color: #b91c1c;">🔴 Тонкие срастания (сульфиды)</div>', unsafe_allow_html=True)
        st.write("Мелкие сульфиды или зерна, значительно замещенные нерудной/темной фазой (магнетитом и др.). Маркер **труднообогатимой руды**.")
    with col_l3:
        st.markdown('<div class="legend-box" style="background-color: rgba(0, 0, 255, 0.2); border-left: 5px solid #0000ff; color: #1d4ed8;">🔵 Тальк</div>', unsafe_allow_html=True)
        st.write("Нежелательная рассеянная фаза в нерудной матрице. Детектируется нейросетью U-Net.")
        
    st.markdown("""
    ---
    ### 🪵 Правила принятия экспертных решений (Decision Tree):
    1. **Проверка оталькования**:
       - Если площадь талька составляет **более 10%** от всей площади шлифа, руда автоматически классифицируется как **Оталькованная** (наиболее критичный фактор обогащения).
    2. **Сравнение сульфидных срастаний** (если талька <= 10%):
       - Алгоритм подсчитывает площадь обычных сульфидов и тонких сульфидов.
       - Если площадь **обычных срастаний** преобладает над тонкими -> руда классифицируется как **Рядовая**.
       - Если площадь **тонких срастаний** преобладает -> руда классифицируется как **Труднообогатимая**.
    """)
