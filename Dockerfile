# Base image with CUDA & PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install basic packages
RUN apt-get update && apt-get install -y \
    git curl wget vim ca-certificates \
    python3.10 python3-pip python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Set default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set working directory
WORKDIR /workspace

# Install Python packages
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers datasets peft accelerate bitsandbytes scipy

# Huggingface token (optional)
ENV HF_HOME="/workspace/.cache/huggingface"

# For LoRA logging / wandb
ENV WANDB_DISABLED=true

# Default command
CMD ["/bin/bash"]