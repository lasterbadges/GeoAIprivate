FROM python:3.10-slim

# Устанавливаем системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости и ставим их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . .

# Открываем стандартный порт Streamlit
EXPOSE 8501

# Команда для правильного запуска Streamlit внутри контейнера
CMD ["streamlit", "run", "GUI.py", "--server.port=8501", "--server.address=0.0.0.0"]
