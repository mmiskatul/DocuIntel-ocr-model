FROM python:3.11-slim

WORKDIR /app

# Install native OS dependencies required for OpenCV and PDF/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user early
RUN useradd -m -u 1000 user

# Configure model caching and environment for unprivileged user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Upgrade pip and install Python packages (suppressing root warnings)
COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore -r requirements.txt

# Pre-cache HuggingFace and RapidOCR weights during build for zero runtime latency
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

# Copy microservice application source code
COPY . .

# Set strict permissions so non-root user owns app files and cached model weights
RUN chown -R user:user /home/user /app

# Switch to non-root user
USER user

# Expose default port
EXPOSE 7860

# Dynamic port binding: binds to Render's $PORT (or 7860 for HuggingFace/local)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
