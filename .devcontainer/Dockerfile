FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=OFF \
    PIP_DEFAULT_TIMEOUT=100 \
    MSYS_NO_PATHCONV=1 \
    PUPPETEER_EXECUTABLE_PATH="/usr/bin/chromium" \
    DEBIAN_FRONTEND=noninteractive

# All system-level installation must run as root.
# The devcontainer base image ends with USER vscode, so we switch back explicitly.
USER root

# Install system dependencies (from .binder/apt.txt + Dockerfile extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ack \
    alien \
    apt-utils \
    bat \
    build-essential \
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
    webp \
    wkhtmltopdf \
    xclip \
    xdg-utils \
    zsh \
    && ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (latest stable) and commonly-used CLI tools
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path && \
    /root/.cargo/bin/cargo install --locked gitui eza tealdeer && \
    cp -a /root/.cargo/bin/* /usr/local/bin/ && \
    chmod 755 /usr/local/bin/cargo
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv and uvx
RUN curl -fsSL https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# Install Typst via the official installer
RUN curl -fsSL https://typst.community/typst-install/install.sh | TYPST_INSTALL_DIR=/usr/local/bin sh || \
    ( curl -fsSL https://github.com/typst/typst/releases/latest/download/typst-x86_64-unknown-linux-musl.tar.xz \
      | tar -xJf - --strip-components=1 -C /usr/local/bin typst-x86_64-unknown-linux-musl/typst \
      && chmod 755 /usr/local/bin/typst )

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
