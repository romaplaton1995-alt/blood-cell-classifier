# Используем официальный образ PyTorch с CUDA 11.8
# Это гарантирует совместимость с большинством GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем их
# Это отдельный слой для эффективного кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/
COPY scripts/ ./scripts/

# Создаём папку для моделей
RUN mkdir -p /app/models

# Устанавливаем права на выполнение скриптов
RUN chmod +x scripts/*.py

# По умолчанию запускаем проверку модели
# Это можно переопределить в docker-compose.yml
CMD ["python", "-c", "import scripts.download_weight; import src.predict; src.predict.check_model()"]