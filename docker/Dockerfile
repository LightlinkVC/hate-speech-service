# Используем официальный образ PyTorch с CUDA
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Кеш для моделей
ENV HF_HOME=/app/model-cache
RUN mkdir -p ${HF_HOME}

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY . .

CMD ["python3", "cmd/main.py"]