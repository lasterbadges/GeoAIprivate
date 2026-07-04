# Используем стабильную версию
FROM python:3.10-slim

# Устанавливаем необходимые системные зависимости
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "GUI.py", "--server.port=8501", "--server.address=0.0.0.0"]
