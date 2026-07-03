import streamlit as st
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="Анализ шлифов руды", page_icon="🔬", layout="wide")

# Инициализируем состояние сессии для кнопки "Выход"
if "logged_out" not in st.session_state:
    st.session_state.logged_out = False

# --- ЭКРАН ВЫХОДА ---
if st.session_state.logged_out:
    st.warning("🔒 Сессия завершена. Вы успешно вышли из системы.")
    st.info("Вы можете безопасно закрыть эту вкладку браузера.")
    if st.button("Войти снова"):
        st.session_state.logged_out = False
        st.rerun()
    st.stop()


# --- ОСНОВНОЙ ИНТЕРФЕЙС ---

# Шапка: Название и кнопка Выхода
col_title, col_exit = st.columns([6, 1])

with col_title:
    st.title("🔬 Модуль визуализации: Анализ шлифов руды")

with col_exit:
    if st.button("🚪 Выход", use_container_width=True):
        st.session_state.logged_out = True
        st.rerun()

st.markdown("---")

# Две колонки для удобного расположения: слева ввод, справа вывод
col_input, col_output = st.columns(2)

with col_input:
    st.subheader("📁 Входные данные")
    
    # Загрузчик файлов
    uploaded_file = st.file_uploader(
        "🖼️ Загрузите фотографию шлифа для определения типа руды", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Кнопка запуска (вынесена из st.form для гибкости)
    run_analysis = st.button("🚀 Определить тип руды", type="primary", use_container_width=True)

with col_output:
    st.subheader("📊 Результаты анализа")
    
    # Логика отрисовки интерфейса при нажатии кнопки
    if run_analysis:
        if uploaded_file is not None:
            # Отображаем загруженное изображение
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Проба: {uploaded_file.name}", use_container_width=True)
            
            # Индикатор загрузки
            with st.spinner('Ожидание ответа от модуля обработки...'):
                
                # --- МЕСТО ДЛЯ МОДУЛЯ ---
                
                
                mock_result_text = "Отобранная (сортированная) руда"
                mock_category = "Рядовая среднезернистая структура"
                # ---------------------------------
            
            # Визуальный вывод результатов
            st.info(f"**Тип руды:** {mock_result_text}")
            
            # Текстовое заключение
            st.markdown(f"""
            **Текстовый отчет:**
            * **Категория структуры:** {mock_category}
            * **Статус:** Успешно обработано внешним модулем.
            
            Данные готовы к экспорту.
            """)
            
        else:
            st.error("⚠️ Ошибка: Сначала загрузите файл изображения во входную колонку.")
    else:
        st.caption("Здесь отобразятся результаты, как только вы загрузите фото и нажмете кнопку.")
