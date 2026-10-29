FROM python:3.11-slim

WORKDIR /app

# Install native OS libraries required for OpenCV and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user early
RUN useradd -m -u 1000 user

# Runtime environment settings
ENV HOME=/home/user \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Upgrade pip and install ultra-lightweight production dependencies (no torch, no sentence-transformers)
COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore -r requirements.txt

# Pre-cache RapidOCR ONNX model for optical character recognition (~15MB only)
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

# Copy microservice application source code
COPY . .

# Set permissions so unprivileged user owns app directory
RUN chown -R user:user /home/user /app

# Switch to non-root user
USER user

# Expose default port
EXPOSE 7860

# Dynamic port binding: binds to Render's $PORT (or 7860 for HuggingFace/local)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
