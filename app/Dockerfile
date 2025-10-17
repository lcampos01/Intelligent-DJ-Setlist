FROM python:3.10-slim

# Dependencias del sistema para Essentia
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libyaml-dev \
    libeigen3-dev \
    python3-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar Essentia desde PyPI (ya compilada)
RUN pip install --no-cache-dir essentia tensorflow==2.15.0

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copiar código
COPY . /app

# crear carpetas de datos
RUN mkdir -p /data /music

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
