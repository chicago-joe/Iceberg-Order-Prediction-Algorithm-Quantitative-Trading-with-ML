# Start with a base Jupyter image with Python 3.12
FROM jupyter/minimal-notebook:python-3.12

USER root

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    cargo \
    xclip \
    ack \
    ripgrep \
    texinfo \
    apt-utils \
    pandoc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Typst via the official installer (avoids snapd issues)
RUN curl -fsSL https://typst.community/typst-install/install.sh | sh && \
    echo 'export PATH="$HOME/.typst:$PATH"' >> /etc/profile && \
    echo 'export PATH="$HOME/.typst:$PATH"' >> ~/.bashrc

# Switch back to the notebook user
USER ${NB_UID}

# Set environment variables
ENV PATH="${HOME}/.typst:${PATH}"

# Copy requirements.txt for Python dependencies
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/

# Install Python packages
RUN pip install --no-cache-dir \
    jupytext \
    myst-parser \
    myst-nb \
    jupyter-book \
    jupyterlab-myst \
    -r /tmp/requirements.txt

# Install JupyterLab Plotly extension
RUN jupyter labextension install jupyterlab-plotly

# Build JupyterLab so extensions take effect
RUN jupyter lab build

# Initialize MyST environment
RUN myst init

# Copy the repository content
COPY --chown=${NB_UID}:${NB_GID} . ${HOME}

# Create build directory
RUN mkdir -p _build

# Set the working directory
WORKDIR ${HOME}

# Convert notebooks to myst if both formats are in repo
RUN jupytext --to myst *.ipynb || echo "No notebooks to convert or conversion failed"

# Pre-build Jupyter-Book
RUN jupyter-book build . || echo "Jupyter-book build will be run on first launch"

# Command to run when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
