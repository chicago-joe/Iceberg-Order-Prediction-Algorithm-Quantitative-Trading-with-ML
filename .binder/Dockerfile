FROM jupyter/base-notebook:python-3.11

# Switch to root for system installations
USER root

# Install system dependencies from apt.txt and additional required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core build tools
    build-essential \
    gcc \
    cmake \
    # Development utilities
    graphviz \
    xclip \
    ack \
    ripgrep \
    alien \
    # Documentation tools
    texinfo \
    apt-utils \
    pandoc \
    # LaTeX and typesetting
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    latexmk \
    # Image processing
    imagemagick \
    webp \
    inkscape \
    # Package management
    rpm \
    # Web and display libraries
    fonts-liberation \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    wkhtmltopdf \
    # Shell and utilities
    zsh \
    curl \
    # Rust dependencies
    pkg-config \
    libssl-dev \
    # Python build dependencies
    python3-dev \
    # Node.js dependencies
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup (latest stable version)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source "$HOME/.cargo/env"' >> ~/.bashrc && \
    export PATH="$HOME/.cargo/bin:$PATH"
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy all Rust binaries to /usr/local/bin to make available for all users
RUN . "$HOME/.cargo/env" && \
    cp -a $HOME/.cargo/bin/* /usr/local/bin/ && \
    chmod 755 /usr/local/bin/cargo

# Install Oh My Zsh, zsh plugins, and set up zsh as default shell for jovyan user
RUN apt-get update && apt-get install -y --no-install-recommends \
    fzf \
    bat \
    cht.sh \
    jq \
    tldr \
    nano \
    && ln -s /usr/bin/batcat /usr/local/bin/bat \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh for jovyan
USER jovyan
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions \
    && mkdir -p ~/bin ~/.local/bin

# Set up zsh as default shell for jovyan user
USER root
RUN chsh -s /bin/zsh jovyan

# Install Typst - note that it installs to /home/jovyan/.typst
RUN curl -fsSL https://typst.community/typst-install/install.sh | sh && \
    # Since it's installed to jovyan's home directory, make it accessible to all
    mkdir -p /usr/local/typst/bin && \
    cp /home/jovyan/.typst/bin/typst /usr/local/typst/bin/ && \
    chmod -R 755 /usr/local/typst && \
    # Add to system PATH
    echo 'export PATH="/usr/local/typst/bin:$PATH"' >> /etc/profile && \
    echo 'export PATH="/usr/local/typst/bin:$PATH"' >> /etc/bash.bashrc

# Install cargo packages system-wide and ensure they're in PATH
RUN . "$HOME/.cargo/env" && \
    cargo install --locked typst-cli gitui eza \
    && cp ~/.cargo/bin/* /usr/local/bin/

# Install uv as root for system-wide availability
RUN curl -fsSL https://astral.sh/uv/install.sh | sh && \
    mv ~/.cargo/bin/uv /usr/local/bin/ && \
    chmod 755 /usr/local/bin/uv

# Set environment variables for Puppeteer
ENV PUPPETEER_EXECUTABLE_PATH="/usr/bin/chromium"
ENV PATH="/usr/local/typst/bin:${PATH}"

# Copy the repository content to the home directory
# This ensures proper ownership of files
COPY --chown=${NB_UID}:${NB_GID} . ${HOME}

# Switch to the repository directory
WORKDIR ${HOME}

# Install npm packages in the repository directory
# This ensures build-components.mjs can find them
RUN npm install --force

# Switch back to jovyan user for remaining operations
USER ${NB_UID}

# Install Python packages using uv
WORKDIR ${HOME}
RUN /usr/local/bin/uv pip install --no-cache-dir -r .binder/requirements.txt

# Install JBang for Java support
RUN mkdir -p ${HOME}/.jbang/bin ${HOME}/.jbang/currentjdk && \
    curl -Ls https://sh.jbang.dev | bash -s - app install --fresh

# Install NVM for Node.js version management
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Create env file with any additional environment variables
RUN mkdir -p ${HOME}/.local/bin && \
    echo '#!/bin/bash' > ${HOME}/.local/bin/env && \
    echo 'echo "Environment loaded"' >> ${HOME}/.local/bin/env && \
    chmod +x ${HOME}/.local/bin/env

# Configure jovyan's zshrc with the exact requested settings
COPY .zshrc ${HOME}/.zshrc

# Ensure tools are available in all shells by adding them to global profile
RUN echo 'export PATH="/usr/local/bin:/usr/local/typst/bin:$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.jbang/bin:$HOME/.jbang/currentjdk/bin:$PATH"' >> /etc/profile && \
    echo 'export PATH="/usr/local/bin:/usr/local/typst/bin:$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.jbang/bin:$HOME/.jbang/currentjdk/bin:$PATH"' >> /etc/bash.bashrc && \
    echo 'source "$HOME/.cargo/env"' >> /etc/profile && \
    echo 'source "$HOME/.cargo/env"' >> /etc/bash.bashrc

# Build components using Node.js
RUN node build-components.mjs || echo "build-components.mjs failed"

# Build JupyterLab using uv
RUN /usr/local/bin/uv run --no-cache jupyter lab build --dev-build=False || echo "jupyter lab build failed"

# Set the working directory to home
WORKDIR ${HOME}

# Ensure all tools are in path when container starts
RUN echo 'source /etc/profile' >> /etc/zsh/zshenv

# Command to run JupyterLab using uv
CMD ["sh", "-c", "/usr/local/bin/uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
