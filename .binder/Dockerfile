# Start with the Jupyter base notebook image with Python 3.9
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
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up zsh as default shell for jovyan user
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

# Install cargo packages system-wide
RUN cargo install --locked typst-cli gitui eza

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

# Set up environment PATH variables for user
RUN echo 'export PATH="/usr/local/typst/bin:${HOME}/.local/bin:${PATH}"' >> ${HOME}/.bashrc && \
    echo 'export PATH="/usr/local/typst/bin:${HOME}/.local/bin:${PATH}"' >> ${HOME}/.zshrc && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ${HOME}/.bashrc && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ${HOME}/.zshrc

# Build components using Node.js
RUN node build-components.mjs || echo "build-components.mjs failed"

# Build JupyterLab using uv
RUN /usr/local/bin/uv run --no-cache jupyter lab build --dev-build=False || echo "jupyter lab build failed"

# Set the working directory to home
WORKDIR ${HOME}

# Command to run JupyterLab using uv
CMD ["sh", "-c", "/usr/local/bin/uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
