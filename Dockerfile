FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CUDA-enabled torch from the cu124 wheel index (matches the base
# image's CUDA 12.4 runtime). Kept separate from requirements.txt so newer
# pip versions don't choke on +local version pins.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# Pre-fetch InsightFace buffalo_l weights so cold starts skip the ~250MB
# download. ctx_id=-1 / CPUExecutionProvider avoids needing CUDA at build time.
RUN python -c "from insightface.app import FaceAnalysis; \
FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], allowed_modules=['detection','recognition'])"

# Pre-fetch YOLO weights into the image so the file is present without being
# tracked in git. Ultralytics downloads to the CWD on first use.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY app ./app

CMD ["python", "-u", "-m", "app.handler"]
