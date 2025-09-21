# Base image with optional CUDA support
ARG BASE_IMAGE=pytorch/pytorch:2.0.1-cpu
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create data directory
RUN mkdir -p data/raw data/processed

# Set environment variables
ENV PYTHONPATH=/app

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Default command
CMD ["python", "scripts/train.py"]
