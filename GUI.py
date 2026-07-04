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
from ore_texture_classifier import TEXTURE_CLASSIFIER_PATH, predict_ore_texture

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

# Путь к U-Net весам талька
MODEL_PATH = "models/test_unet.pth"
RESNET_MODEL_PATH = "models/ore_resnet18.pth"
TEXTURE_MODEL_PATH = str(TEXTURE_CLASSIFIER_PATH)
FEEDBACK_CSV = "data/geologist_feedback.csv"
ACTIVE_LEARNING_DIR = "data/active_learning"
HISTORY_CSV = "data/analysis_history.csv"
MAX_BATCH_PREVIEWS = 12
BATCH_INPUT_PREVIEW_SIDE = 420
BATCH_RESULT_PREVIEW_SIDE = 760

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
.batch-result-shell {
    margin-top: 18px;
    padding: 1px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(16,185,129,0.55), rgba(59,130,246,0.45), rgba(245,158,11,0.35));
}
.batch-result-inner {
    border-radius: 17px;
    padding: 18px;
    background: rgba(8, 13, 24, 0.94);
    border: 1px solid rgba(255,255,255,0.08);
}
.result-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 12px;
}
.result-file-name {
    color: #f8fafc;
    font-size: 16px;
    font-weight: 700;
    overflow-wrap: anywhere;
}
.result-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    color: #07111f;
    background: #67e8f9;
    white-space: nowrap;
}
.result-mini-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 8px 0 16px 0;
}
.result-mini-cell {
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.07);
}
.result-mini-label {
    font-size: 11px;
    color: #94a3b8;
    font-weight: 700;
    text-transform: uppercase;
}
.result-mini-value {
    margin-top: 3px;
    color: #f8fafc;
    font-size: 16px;
    font-weight: 800;
}
@media (max-width: 900px) {
    .metrics-container,
    .result-mini-grid {
        display: grid;
        grid-template-columns: 1fr;
    }
    .result-title-row {
        align-items: flex-start;
        flex-direction: column;
    }
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


def load_history_safely(filepath):
    import csv
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=[
            "Timestamp", "Filename", "Class", "TalcPercent", "SulfidePercent", 
            "OrdinaryPercent", "FinePercent", "TextureClass", "TextureConfidence",
            "ResNetClass", "ResNetConfidence", "EnsembleClass", "EnsembleConfidence"
        ])
    
    target_cols = [
        "Timestamp", "Filename", "Class", "TalcPercent", "SulfidePercent", 
        "OrdinaryPercent", "FinePercent", "TextureClass", "TextureConfidence",
        "ResNetClass", "ResNetConfidence", "EnsembleClass", "EnsembleConfidence"
    ]
    
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            for row in reader:
                if not row:
                    continue
                # Если в строке меньше 13 колонок, дополняем пустыми строками
                if len(row) < 13:
                    row = row + [""] * (13 - len(row))
                elif len(row) > 13:
                    row = row[:13]
                rows.append(row)
        
        df = pd.DataFrame(rows, columns=target_cols)
        for col in ["TalcPercent", "SulfidePercent", "OrdinaryPercent", "FinePercent"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        return pd.DataFrame(columns=target_cols)


# --- ОСНОВНАЯ ЛОГИКА АНАЛИЗА ---
def analyze_single_image(img_path, brightness_thresh, talc_thresh, use_unet, use_resnet=True):
    # 1. Читаем и предобрабатываем изображение (оно автоматически уменьшено до max_side=3200)
    image_bgr, image_enhanced_gray, image_hsv = pre_pr.preprocess_image(img_path)
    h_orig, w_orig = image_bgr.shape[:2]
    
    # 2. Создаем бинарную маску
    filled_mask = pre_pr.create_binary_mask(image_hsv)
        
    # 3. Запускаем классификацию сегментов
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
        
    # Текстурная классификация
    texture_result = None
    texture_confidence = 0.0
    regular_probability = 0.0
    complex_probability = 0.0
    texture_tile_count = 0
    try:
        texture_result = predict_ore_texture(image_bgr, TEXTURE_MODEL_PATH)
        texture_class = texture_result["class"]
        texture_confidence = texture_result["confidence"]
        regular_probability = texture_result["regular_probability"]
        complex_probability = texture_result["complex_probability"]
        texture_tile_count = texture_result["tile_count"]
    except Exception:
        texture_class = "Рядовая" if ordinary_pixels >= fine_pixels else "Труднообогатимая"
        texture_confidence = max(ordinary_pct, fine_pct)
        regular_probability = ordinary_pct
        complex_probability = fine_pct

    resnet_class = "Рядовая" if ordinary_pct >= fine_pct else "Труднообогатимая"
    resnet_confidence = max(ordinary_pct, fine_pct)
    resnet_regular_probability = ordinary_pct
    resnet_complex_probability = fine_pct

    if sulfide_pixels > 0:
        texture_weight = 0.6 if texture_result is not None else 0.0
        resnet_weight = 1.0 - texture_weight
        ensemble_regular_probability = (
            regular_probability * texture_weight
            + resnet_regular_probability * resnet_weight
        )
    else:
        ensemble_regular_probability = regular_probability

    ensemble_complex_probability = 100.0 - ensemble_regular_probability
    ensemble_class = "Рядовая" if ensemble_regular_probability >= 50.0 else "Труднообогатимая"
    ensemble_confidence = max(ensemble_regular_probability, ensemble_complex_probability)

    if talc_pct > 10.0:
        ore_class = "Оталькованная"
    else:
        ore_class = ensemble_class
        
    # Для отображения в Streamlit сжимаем RGB оригинал и маску до max_side_display=1600
    max_side_display = 1600
    if max(h_orig, w_orig) > max_side_display:
        disp_scale = max_side_display / float(max(h_orig, w_orig))
        disp_w = int(w_orig * disp_scale)
        disp_h = int(h_orig * disp_scale)
        original_rgb_disp = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        final_mask_disp = cv2.resize(final_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    else:
        original_rgb_disp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        final_mask_disp = final_mask
        
    # Создаем RGB маску отображения
    h_disp, w_disp = final_mask_disp.shape[:2]
    mask_rgb_disp = np.zeros((h_disp, w_disp, 3), dtype=np.uint8)
    mask_rgb_disp[final_mask_disp == 1] = [0, 255, 0]   # Зеленый
    mask_rgb_disp[final_mask_disp == 2] = [255, 0, 0]   # Красный
    mask_rgb_disp[final_mask_disp == 3] = [0, 0, 255]   # Синий
    
    return {
        "class": ore_class,
        "talc_pct": talc_pct,
        "sulfide_pct": sulfide_pct,
        "ordinary_pct": ordinary_pct,
        "fine_pct": fine_pct,
        "texture_class": texture_class,
        "texture_confidence": texture_confidence,
        "regular_probability": regular_probability,
        "complex_probability": complex_probability,
        "texture_tile_count": texture_tile_count,
        "texture_model_used": texture_result is not None,
        "resnet_class": resnet_class,
        "resnet_confidence": round(resnet_confidence, 1),
        "resnet_regular_probability": round(resnet_regular_probability, 1),
        "resnet_complex_probability": round(resnet_complex_probability, 1),
        "ensemble_class": ensemble_class,
        "ensemble_confidence": round(ensemble_confidence, 1),
        "ensemble_regular_probability": round(ensemble_regular_probability, 1),
        "ensemble_complex_probability": round(ensemble_complex_probability, 1),
        "original_rgb": original_rgb_disp,
        "mask_rgb": mask_rgb_disp,
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
        "FinePercent": res["fine_pct"],
        "TextureClass": res.get("texture_class", ""),
        "TextureConfidence": res.get("texture_confidence", 0.0),
        "ResNetClass": res.get("resnet_class", ""),
        "ResNetConfidence": res.get("resnet_confidence", 0.0),
        "EnsembleClass": res.get("ensemble_class", ""),
        "EnsembleConfidence": res.get("ensemble_confidence", 0.0)
    }])
    
    if os.path.exists(HISTORY_CSV):
        try:
            df.to_csv(HISTORY_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
        except Exception:
            pass
    else:
        df.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')


def resize_rgb_for_display(image_rgb, max_side=1100, interpolation=cv2.INTER_AREA):
    h, w = image_rgb.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image_rgb

    scale = max_side / float(side)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image_rgb, new_size, interpolation=interpolation)


def make_overlay(original_rgb, mask_rgb, alpha):
    return (original_rgb * (1.0 - alpha) + mask_rgb * alpha).astype(np.uint8)


def render_before_after(original_rgb, mask_rgb, alpha, label_prefix="", max_side=1100):
    original_small = resize_rgb_for_display(original_rgb, max_side=max_side)
    mask_small = resize_rgb_for_display(mask_rgb, max_side=max_side, interpolation=cv2.INTER_NEAREST)
    overlay_small = make_overlay(original_small, mask_small, alpha)

    if HAS_IMAGE_COMPARISON:
        image_comparison(
            img1=Image.fromarray(original_small),
            img2=Image.fromarray(overlay_small),
            label1=f"{label_prefix}Оригинал",
            label2=f"{label_prefix}Маска",
            show_labels=True,
            make_responsive=True,
            starting_position=50
        )
    else:
        st.image(overlay_small, caption=f"{label_prefix}Маска наложения", use_container_width=True)


def make_display_pair(original_rgb, mask_rgb, alpha=0.42, max_side=BATCH_RESULT_PREVIEW_SIDE):
    original_small = resize_rgb_for_display(original_rgb, max_side=max_side)
    mask_small = resize_rgb_for_display(mask_rgb, max_side=max_side, interpolation=cv2.INTER_NEAREST)
    overlay_small = make_overlay(original_small, mask_small, alpha)
    return original_small, mask_small, overlay_small


def load_preview_image(file_obj, max_side=BATCH_INPUT_PREVIEW_SIDE):
    file_obj.seek(0)
    image = Image.open(file_obj).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    file_obj.seek(0)
    return image


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
    col_input, col_preview = st.columns([1, 1])

    with col_input:
        st.subheader("📁 Входные данные")
        uploaded_file = st.file_uploader(
            "🖼️ Загрузите микрофотографию шлифа",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "tga"]
        )
        use_resnet = True
        brightness_thresh = 120
        talc_thresh = 30
        use_unet = True
        opacity_slider = st.slider(
            "Прозрачность маски наложения (%)",
            min_value=0, max_value=100, value=40, step=5
        )
        run_analysis = st.button("🚀 Начать анализ", type="primary", use_container_width=True)

    with col_preview:
        st.subheader("🖼️ Превью")
        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), caption=uploaded_file.name, use_container_width=True)
            uploaded_file.seek(0)
        else:
            st.info("💡 Загрузите изображение слева и нажмите «Начать анализ».")

    # --- результаты на полной ширине ---
    if run_analysis and uploaded_file is not None:
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        uploaded_file.seek(0)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Анализируем микроструктуру..."):
            start_time = time.time()
            res = analyze_single_image(str(temp_path), brightness_thresh, talc_thresh, use_unet, use_resnet)
            duration = time.time() - start_time
            save_analysis_to_history(uploaded_file.name, res)

        st.markdown("---")

        # баннер сорта руды
        if res["class"] == "Оталькованная":
            st.markdown(f"""
            <div class="ore-banner-card ore-talc">
                <div class="ore-banner-title">Рекомендуемый сорт руды</div>
                <div class="ore-banner-value">ОТАЛЬКОВАННАЯ РУДА</div>
                <div class="ore-banner-desc">Содержание талька превышает критический порог 10% (тальк — {res['talc_pct']}%). Требуется введение депрессоров талька в цикле флотации.</div>
            </div>
            """, unsafe_allow_html=True)
        elif res["class"] == "Труднообогатимая":
            st.markdown(f"""
            <div class="ore-banner-card ore-difficult">
                <div class="ore-banner-title">Рекомендуемый сорт руды</div>
                <div class="ore-banner-value">ТРУДНООБОГАТИМАЯ РУДА</div>
                <div class="ore-banner-desc">Доля талька в норме (тальк — {res['talc_pct']}%). Общий голос ResNet-маски и текстурной модели относит снимок к труднообогатимой руде с уверенностью {res['ensemble_confidence']}%.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ore-banner-card ore-ordinary">
                <div class="ore-banner-title">Рекомендуемый сорт руды</div>
                <div class="ore-banner-value">РЯДОВАЯ РУДА</div>
                <div class="ore-banner-desc">Доля талька в норме (тальк — {res['talc_pct']}%). Общий голос ResNet-маски и текстурной модели относит снимок к рядовой руде с уверенностью {res['ensemble_confidence']}%.</div>
            </div>
            """, unsafe_allow_html=True)

        # слайдер сравнения (полная ширина)
        alpha = opacity_slider / 100.0
        render_before_after(res["original_rgb"], res["mask_rgb"], alpha)
        st.caption(f"Перетащите бегунок для сравнения оригинала и маски · Время анализа: {duration:.2f} сек.")

        # метрики
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
                <div class="metric-label">ResNet по маске</div>
                <div class="metric-value">{res['resnet_regular_probability']}% / {res['resnet_complex_probability']}%</div>
                <div class="metric-sublabel">Рядовая / труднообогатимая</div>
            </div>
            <div class="metric-glass-card">
                <div class="metric-label">Texture classifier</div>
                <div class="metric-value">{res['regular_probability']}% / {res['complex_probability']}%</div>
                <div class="metric-sublabel">Рядовая / труднообогатимая · {res['texture_tile_count']} тайлов</div>
            </div>
            <div class="metric-glass-card">
                <div class="metric-label">Общий итог</div>
                <div class="metric-value">{res['ensemble_regular_probability']}% / {res['ensemble_complex_probability']}%</div>
                <div class="metric-sublabel">Ансамбль двух моделей</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # таблица
        df_metrics = pd.DataFrame({
            "Параметр": ["Сорт руды", "Доля талька", "Общая доля сульфидов",
                         "Текстурный сорт", "Уверенность классификатора"],
            "Значение": [res["class"].upper(), f"{res['talc_pct']}%", f"{res['sulfide_pct']}%",
                         res["texture_class"].upper(), f"{res['texture_confidence']}%"]
        })
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        report_text = (
            f"Отчет анализа шлифа {uploaded_file.name}\n"
            f"Сорт руды: {res['class'].upper()}\n"
            f"Тальк: {res['talc_pct']}%\n"
            f"Сульфиды: {res['sulfide_pct']}%\n"
            f"Текстурный сорт: {res['texture_class'].upper()}\n"
            f"Уверенность классификатора: {res['texture_confidence']}%\n"
        )
        st.download_button(
            "📥 Скачать текстовый отчет",
            data=report_text,
            file_name=f"Report_{uploaded_file.name.split('.')[0]}.txt",
            mime="text/plain"
        )

        # экспертная верификация
        st.markdown("---")
        st.subheader("✍️ Экспертная верификация (Active Learning)")
        st.markdown("Если не согласны с решением алгоритма — укажите верный сорт. Файл запишется в обучающую выборку.")
        corrected_class = st.selectbox(
            "Фактический сорт руды:",
            ["Рядовая", "Труднообогатимая", "Оталькованная"],
            index=["Рядовая", "Труднообогатимая", "Оталькованная"].index(res["class"])
        )
        if st.button("💾 Сохранить оценку и отправить в обучающий пул"):
            save_geologist_feedback(
                uploaded_file.name, res["class"], corrected_class,
                res["talc_pct"], res["sulfide_pct"], str(temp_path)
            )
            st.success("✅ Отзыв сохранён! Данные отправлены в базу дообучения моделей.")

    elif run_analysis:
        st.error("⚠️ Сначала выберите изображение шлифа для анализа.")


# --- ВКЛАДКА 2: ПАКЕТНЫЙ АНАЛИЗ ---
with tab2:
    st.subheader("📦 Обработка партий шлифов")
    st.caption("Откройте нужную папку в диалоге и нажмите Ctrl+A — выберутся все файлы сразу.")

    batch_files = st.file_uploader(
        "Выберите изображения для пакетной обработки:",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "tga"],
        accept_multiple_files=True
    )

    b_brightness_thresh = 120
    b_talc_thresh = 30
    b_use_unet = True
    b_use_resnet = True

    start_batch = st.button("⚡ Начать обработку серии", type="primary")
    batch_preview_area = st.empty()

    if batch_files and not start_batch:
        with batch_preview_area.container():
            st.markdown(f"**{len(batch_files)} файлов выбрано** — превью до анализа:")
            _preview_cols = st.columns(4)
            for _pi, _pf in enumerate(batch_files[:MAX_BATCH_PREVIEWS]):
                with _preview_cols[_pi % 4]:
                    st.image(
                        load_preview_image(_pf),
                        caption=_pf.name[:22] + ("…" if len(_pf.name) > 22 else ""),
                        use_container_width=True
                    )
            if len(batch_files) > MAX_BATCH_PREVIEWS:
                st.caption(f"Показаны первые {MAX_BATCH_PREVIEWS} превью, остальные файлы тоже будут обработаны.")

    if start_batch and batch_files:
        results = []
        result_previews = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        batch_preview_area.empty()

        for idx, _f in enumerate(batch_files):
            status_text.text(f"Обработка {idx+1}/{len(batch_files)}: {_f.name}")
            _tmp = temp_dir / _f.name
            with open(_tmp, "wb") as _fh:
                _fh.write(_f.getbuffer())
            res = analyze_single_image(str(_tmp), b_brightness_thresh, b_talc_thresh, b_use_unet, b_use_resnet)
            save_analysis_to_history(_f.name, res)
            if len(result_previews) < MAX_BATCH_PREVIEWS:
                original_preview, mask_preview, _ = make_display_pair(res["original_rgb"], res["mask_rgb"], alpha=0.42)
                result_previews.append({
                    "name": _f.name,
                    "class": res["class"],
                    "talc_pct": res["talc_pct"],
                    "texture_confidence": res["texture_confidence"],
                    "original": original_preview,
                    "mask": mask_preview,
                })
            results.append({
                "Имя файла": _f.name,
                "Рекомендуемый сорт": res["class"].upper(),
                "Тальк (%)": res["talc_pct"],
                "Сульфиды (%)": res["sulfide_pct"],
                "Текстурный сорт": res["texture_class"].upper(),
                "Уверенность (%)": res["texture_confidence"]
            })
            del res
            progress_bar.progress((idx + 1) / len(batch_files))

        status_text.text("✅ Обработка серии завершена!")
        with batch_preview_area.container():
            st.markdown("**Превью после анализа — оригинал ↔ маска:**")
            st.caption("Это тот же блок, где до запуска были обычные превью. Теперь здесь слайдеры результата.")
            if len(batch_files) > MAX_BATCH_PREVIEWS:
                st.caption(f"Показаны первые {MAX_BATCH_PREVIEWS} результатов, вся партия сохранена в таблице ниже.")
            for _pi, item in enumerate(result_previews):
                with st.container():
                    st.markdown(
                        f"**{item['name']}**  \n"
                        f"{item['class'].upper()} · тальк {item['talc_pct']}% · уверенность {item['texture_confidence']}%"
                    )
                    render_before_after(item["original"], item["mask"], 0.42, max_side=BATCH_RESULT_PREVIEW_SIDE)

        df_batch = pd.DataFrame(results)
        st.dataframe(df_batch, use_container_width=True, hide_index=True)
        csv_data = df_batch.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Скачать сводный отчет (CSV)",
            data=csv_data,
            file_name="batch_report_ore_classification.csv",
            mime="text/csv"
        )

    elif start_batch:
        st.error("⚠️ Загрузите хотя бы один файл для пакетного анализа.")


# --- ВКЛАДКА 3: АНАЛИТИЧЕСКИЙ ДАШБОРД ---
with tab3:
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 26px; font-weight: 800; margin: 0;
        ">📊 Аналитический дашборд</h2>
        <p style="color: #9ca3af; font-size: 14px; margin: 4px 0 0 0;">
            Статистика по всем проанализированным шлифам в этой сессии
        </p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(HISTORY_CSV):
        df_hist = load_history_safely(HISTORY_CSV)

        total = len(df_hist)
        cnt_ordinary = int((df_hist["Class"] == "Рядовая").sum())
        cnt_difficult = int((df_hist["Class"] == "Труднообогатимая").sum())
        cnt_talc = int((df_hist["Class"] == "Оталькованная").sum())
        avg_talc = round(df_hist["TalcPercent"].mean(), 2)
        avg_sulfide = round(df_hist["SulfidePercent"].mean(), 2)

        # KPI карточки
        st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-glass-card">
                <div class="metric-label">Всего шлифов</div>
                <div class="metric-value">{total}</div>
                <div class="metric-sublabel">Всего обработано</div>
            </div>
            <div class="metric-glass-card">
                <div class="metric-label">Рядовая руда</div>
                <div class="metric-value" style="color:#10b981;">{cnt_ordinary}</div>
                <div class="metric-sublabel">{round(cnt_ordinary/total*100,1) if total else 0}% от общего</div>
            </div>
            <div class="metric-glass-card">
                <div class="metric-label">Труднообогатимая</div>
                <div class="metric-value" style="color:#f59e0b;">{cnt_difficult}</div>
                <div class="metric-sublabel">{round(cnt_difficult/total*100,1) if total else 0}% от общего</div>
            </div>
            <div class="metric-glass-card">
                <div class="metric-label">Оталькованная</div>
                <div class="metric-value" style="color:#6366f1;">{cnt_talc}</div>
                <div class="metric-sublabel">{round(cnt_talc/total*100,1) if total else 0}% от общего</div>
            </div>
            <div class="metric-glass-card">
                <div class="metric-label">Ср. тальк / сульфиды</div>
                <div class="metric-value">{avg_talc}% / {avg_sulfide}%</div>
                <div class="metric-sublabel">Среднее по всем шлифам</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_pie, col_bar = st.columns(2)

        with col_pie:
            class_counts = df_hist["Class"].value_counts().reset_index()
            class_counts.columns = ["Сорт руды", "Кол-во"]
            color_map = {
                "Рядовая": "#10b981",
                "Труднообогатимая": "#f59e0b",
                "Оталькованная": "#6366f1"
            }
            fig_pie = px.pie(
                class_counts,
                names="Сорт руды",
                values="Кол-во",
                color="Сорт руды",
                color_discrete_map=color_map,
                hole=0.55,
                title="Распределение сортов руды"
            )
            fig_pie.update_traces(
                textposition='outside',
                textinfo='percent+label',
                pull=[0.05] * len(class_counts),
                marker=dict(line=dict(color='#0b0f19', width=2))
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(15,23,42,0.6)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Inter'),
                title_font=dict(size=15, color='#f3f4f6'),
                showlegend=False,
                margin=dict(t=50, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_hist["TalcPercent"],
                name="Тальк",
                marker_color='#6366f1',
                opacity=0.8,
                nbinsx=20
            ))
            fig_hist.add_trace(go.Histogram(
                x=df_hist["SulfidePercent"],
                name="Сульфиды",
                marker_color='#10b981',
                opacity=0.8,
                nbinsx=20
            ))
            fig_hist.update_layout(
                barmode='overlay',
                title="Распределение содержания минералов (%)",
                paper_bgcolor='rgba(15,23,42,0.6)',
                plot_bgcolor='rgba(15,23,42,0.3)',
                font=dict(color='#cbd5e1', family='Inter'),
                title_font=dict(size=15, color='#f3f4f6'),
                xaxis=dict(title="Содержание (%)", gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title="Кол-во шлифов", gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1')),
                margin=dict(t=50, b=40, l=40, r=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Динамика по времени (содержание талька)
        if total >= 3:
            df_hist["Timestamp"] = pd.to_datetime(df_hist["Timestamp"])
            df_hist_sorted = df_hist.sort_values("Timestamp")

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_hist_sorted["Timestamp"],
                y=df_hist_sorted["TalcPercent"],
                mode='lines+markers',
                name='Тальк %',
                line=dict(color='#6366f1', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(99,102,241,0.1)'
            ))
            fig_line.add_trace(go.Scatter(
                x=df_hist_sorted["Timestamp"],
                y=df_hist_sorted["SulfidePercent"],
                mode='lines+markers',
                name='Сульфиды %',
                line=dict(color='#10b981', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(16,185,129,0.1)'
            ))
            # Пороговая линия талька
            fig_line.add_hline(
                y=10.0, line_dash="dot",
                line_color="#ef4444",
                annotation_text="Порог талька 10%",
                annotation_position="bottom right",
                annotation_font_color="#ef4444"
            )
            fig_line.update_layout(
                title="Динамика минерального состава по времени",
                paper_bgcolor='rgba(15,23,42,0.6)',
                plot_bgcolor='rgba(15,23,42,0.3)',
                font=dict(color='#cbd5e1', family='Inter'),
                title_font=dict(size=15, color='#f3f4f6'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title="Содержание (%)", gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1')),
                margin=dict(t=50, b=40, l=40, r=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # Таблица последних 10 шлифов
        st.markdown("<h4 style='color:#f3f4f6; margin-top:10px;'>🕐 Последние 10 обработанных шлифов</h4>", unsafe_allow_html=True)
        st.dataframe(
            df_hist.tail(10)[["Timestamp", "Filename", "Class", "TalcPercent", "SulfidePercent", "OrdinaryPercent", "FinePercent"]].iloc[::-1].rename(columns={
                "Timestamp": "Время", "Filename": "Файл", "Class": "Сорт",
                "TalcPercent": "Тальк %", "SulfidePercent": "Сульфиды %",
                "OrdinaryPercent": "Обычные %", "FinePercent": "Тонкие %"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Кнопка очистки истории
        if st.button("🗑️ Очистить историю анализов", type="secondary"):
            os.remove(HISTORY_CSV)
            st.success("История очищена!")
            st.rerun()

        # Скачать историю
        csv_hist = df_hist.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Скачать полную историю (CSV)",
            data=csv_hist,
            file_name="analysis_history.csv",
            mime="text/csv"
        )
    else:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 80px 20px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            margin-top: 30px;
        ">
            <div style="font-size: 64px; margin-bottom: 20px;">📭</div>
            <h3 style="color: #f3f4f6; font-weight: 700; margin-bottom: 10px;">История пуста</h3>
            <p style="color: #9ca3af; font-size: 14px;">
                Проанализируйте несколько шлифов во вкладке «Анализ одного снимка»<br>
                или загрузите серию через «Пакетный анализ» — данные появятся здесь автоматически.
            </p>
        </div>
        """, unsafe_allow_html=True)


# --- ВКЛАДКА 4: МЕТОДОЛОГИЯ ---
with tab4:
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
