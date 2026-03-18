FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=OFF \
    PIP_DEFAULT_TIMEOUT=100 \
    MSYS_NO_PATHCONV=1 \
    PUPPETEER_EXECUTABLE_PATH="/usr/bin/chromium"

# Install system dependencies (from .binder/apt.txt + Dockerfile extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ack \
    alien \
    apt-utils \
    bat \
    build-essential \
    cargo \
    cmake \
    curl \
    fzf \
    fonts-liberation \
    gcc \
    graphviz \
    imagemagick \
    inkscape \
    jq \
    latexmk \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libssl-dev \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    nano \
    nodejs \
    npm \
    pandoc \
    pkg-config \
    python3-dev \
    ripgrep \
    rpm \
    texinfo \
    texlive-fonts-recommended \
    texlive-plain-generic \
    texlive-xetex \
    tldr \
    webp \
    wkhtmltopdf \
    xclip \
    xdg-utils \
    zsh \
    && ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (latest stable) and make binaries available system-wide
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH" && \
    . "$HOME/.cargo/env" && \
    cargo install --locked gitui eza && \
    cp -a "$HOME/.cargo/bin/"* /usr/local/bin/ && \
    chmod 755 /usr/local/bin/cargo
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv and uvx
RUN curl -fsSL https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx && \
    chmod 755 /usr/local/bin/uv /usr/local/bin/uvx

# Install Typst
RUN curl -fsSL https://typst.community/typst-install/install.sh | sh && \
    mkdir -p /usr/local/typst/bin && \
    cp /root/.typst/bin/typst /usr/local/typst/bin/ && \
    chmod -R 755 /usr/local/typst
ENV PATH="/usr/local/typst/bin:${PATH}"

# Install Oh My Zsh + plugins for vscode user
USER vscode
ENV ZSH="/home/vscode/.oh-my-zsh"
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions \
        "${ZSH_CUSTOM:-/home/vscode/.oh-my-zsh/custom}/plugins/zsh-autosuggestions" && \
    git clone https://github.com/zsh-users/zsh-completions \
        "${ZSH_CUSTOM:-/home/vscode/.oh-my-zsh/custom}/plugins/zsh-completions" && \
    mkdir -p /home/vscode/bin /home/vscode/.local/bin

# Install NVM for vscode user
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

USER root
# Set zsh as default shell for vscode user
RUN chsh -s /bin/zsh vscode

WORKDIR /workspaces
