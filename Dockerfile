FROM python:3.11-slim

WORKDIR /app

# Create unprivileged application user
RUN useradd -m -u 1000 user

# Pure Cloud API environment
ENV HOME=/home/user \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Install ultra-lightweight pure API requirements (zero local models, zero torch, zero onnx)
COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore -r requirements.txt

# Copy microservice application source code
COPY . .

# Set permissions for application user
RUN chown -R user:user /home/user /app

USER user

EXPOSE 7860

# Dynamic port binding: binds to Render's $PORT (or 7860 for HuggingFace/local)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
