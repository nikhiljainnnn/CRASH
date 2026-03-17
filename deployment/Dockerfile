# Base image with CUDA 12.1 and Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set non-interactive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Python, pip, and OpenGL requirements for OpenCV/PyTorch
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory inside the container
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
# Install specific ML serving and monitoring libraries
RUN pip3 install --no-cache-dir -r requirements.txt \
    fastapi uvicorn prometheus-client onnxruntime-gpu

# Copy all application code
COPY . .

# Expose FastAPI and Prometheus metrics ports
EXPOSE 8000
EXPOSE 8001

# Run the inference server
CMD ["python3", "deployment/app.py"]
