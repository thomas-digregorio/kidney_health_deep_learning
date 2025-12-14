# Use official RAPIDS image as base (Ubuntu + CUDA + Python + RAPIDS)
# Matches env requirements: Python 3.12, CUDA 12.x, RAPIDS 24.10
FROM rapidsai/base:24.10-cuda12.5-py3.12

# Set working directory
WORKDIR /app

# Switch to root to install system dependencies if needed
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to default user (usually 'rapids' or 'mambauser' in these images)
# But for simplicity in many DS workflows, we stick to the default provided by the image
# The rapids image uses 'mambauser' usually.
USER mambauser

# Copy environment file
COPY --chown=mambauser:mambauser environment.yaml /app/environment.yaml

# Install dependencies
# We use mamba (included in base) for speed
RUN mamba env update -n base -f /app/environment.yaml && \
    mamba clean --all -f -y

# Copy source code
COPY --chown=mambauser:mambauser src/ /app/src/
COPY --chown=mambauser:mambauser .env /app/.env

# Default command
CMD ["python", "src/train_ckd.py"]
