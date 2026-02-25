FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make cmake git libopenblas-dev ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install llama-cpp-python from pre-built wheel (this is the key fix - no compilation!)
RUN pip install --no-cache-dir \
    https://huggingface.co/Luigi/llama-cpp-python-wheels-hf-spaces-free-cpu/resolve/main/llama_cpp_python-0.3.22-cp310-cp310-linux_x86_64.whl

# Install other dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    gradio==5.12.0 \
    sentence-transformers>=3.0.0 \
    scikit-learn>=1.5.0 \
    numpy>=1.26.0 \
    huggingface_hub>=0.25.0

# Copy application code
COPY . .

# Create a non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /home/user/app
COPY --chown=user . /home/user/app

EXPOSE 7860

CMD ["python", "app.py"]
