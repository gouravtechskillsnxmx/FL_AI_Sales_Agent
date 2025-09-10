# Dockerfile - Python app with ffmpeg for Twilio Media Streams
# - Uses python:3.11-slim for a small image
# - Installs ffmpeg from apt
# - Installs pip requirements
# - Starts app with gunicorn + uvicorn workers
# - Exposes $PORT (default 8000)
FROM python:3.11-slim

# Avoid prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system deps (ffmpeg + build deps)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       gcc \
       libsndfile1 \
       curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Create a non-root user and chown files (optional but good practice)
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose the port
EXPOSE ${PORT}

# Simple healthcheck (checks /health endpoint)
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Start command (uses $PORT if set; fallback to 8000)
# Render supports using this Dockerfile directly. The command expands PORT at runtime.
CMD ["/bin/sh", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker langchain_agent_outbound:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120"]
