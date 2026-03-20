FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive

# The devcontainer base image ends with USER vscode — switch back to root
# for all system-level installations.
USER root

# Minimal system deps required to build/run the Python ML stack:
#   - build-essential / gcc / python3-dev / pkg-config  → compile C extensions
#   - libssl-dev                                         → cryptography package
#   - libhdf5-dev                                        → tables (PyTables)
#   - graphviz                                           → graphviz Python package
#   - pandoc                                             → nbconvert / myst
#   - fonts-liberation + X11/GTK libs                   → matplotlib rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    fonts-liberation \
    gcc \
    graphviz \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libhdf5-dev \
    libssl-dev \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    pandoc \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv — fast Python package manager used in on-create.sh
RUN curl -fsSL https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

USER vscode
WORKDIR /workspaces
