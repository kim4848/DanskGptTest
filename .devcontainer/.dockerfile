# Base image
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and Intel Extension for PyTorch
RUN pip install --no-cache-dir torch==1.13.1 torchvision torchaudio \
    intel_extension_for_pytorch

# Set the working directory
WORKDIR /workspace

# Expose ports if necessary
EXPOSE 8888
