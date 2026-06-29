FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip git git-lfs curl wget \
    build-essential libssl-dev libffi-dev && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/moa-coord
COPY requirements.txt .

RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install torch==2.3.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 && \
    python3.12 -m pip install -r requirements.txt

COPY . .
ENV PYTHONPATH=/workspace/moa-coord
CMD ["bash"]
