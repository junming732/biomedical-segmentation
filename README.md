# Biomedical Image Segmentation

This repository contains code for biomedical image segmentation using U‑Net and ResUNet architectures.
It is based on an assignment from **Advanced Deep Learning for Image Processing (1MD042)**.

## Project structure

```
biomedical-segmentation/
├── models/                  # Model definitions (UNet, ResUNet)
├── scripts/                 # Training scripts
├── utils/                   # Data loading & preprocessing
├── tests/                   # Unit tests (pytest)
├── pyproject.toml           # Project config (dependencies, tools)
├── .pre-commit-config.yaml  # Local hooks (flake8, pytest, etc.)
└── .github/workflows/ci.yml # GitHub Actions workflow
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -U pip

# Install the project and its dependencies
pip install -e .

```

## Pre-commit (recommended)

```bash
pre-commit install
pre-commit run --all-files
```

## Running tests

```bash
pytest -q
```

## Training

```bash
python scripts/train.py --config configs/unet.yaml
```

## Docker

This repo includes Docker support (e.g., a `Dockerfile` and possibly `docker-compose.yml`).

### Build and run with Dockerfile
```bash
# Build image (use the Dockerfile in the repo root; adjust -f if it lives elsewhere)
docker build -t biomedical-segmentation -f Dockerfile .

# Run tests inside the container
docker run --rm -it -v "$PWD":/app -w /app biomedical-segmentation pytest -q

# Train inside the container
docker run --rm -it -v "$PWD":/app -w /app biomedical-segmentation \
  python scripts/train.py --config configs/unet.yaml
```

### Using Docker Compose (if `docker-compose.yml` / `compose.yaml` is present)
```bash
# Build and run tests
docker compose run --rm app pytest -q

# Train
docker compose run --rm app python scripts/train.py --config configs/unet.yaml
```
> Replace `app` with the actual service name in your compose file if different.

## Continuous Integration (GitHub Actions)

A minimal CI workflow is provided at `.github/workflows/ci.yml`. It installs the project from
`pyproject.toml`
