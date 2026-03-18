#!/bin/bash
# post-create.sh — runs after EACH codespace is created (not during prebuild)
# Use for per-session, user-specific, or secret-dependent setup.
set -euo pipefail

echo "==> [post-create] Starting post-creation setup"

REPO_ROOT="${CODESPACE_VSCODE_FOLDER:-/workspaces/$(basename "$(pwd)")}"
cd "$REPO_ROOT"

# ── Shell configuration ────────────────────────────────────────────────────
ZSH_HOME="/home/vscode/.oh-my-zsh"
ZSHRC="/home/vscode/.zshrc"

if [ -f "$ZSH_HOME/oh-my-zsh.sh" ]; then
    echo "==> [post-create] Writing .zshrc"
    cat > "$ZSHRC" <<'ZSHRC_EOF'
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(
  git
  zsh-autosuggestions
  zsh-completions
)
fpath+=${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions/src
source $ZSH/oh-my-zsh.sh

export PATH="$PATH:$HOME/bin/"
export UV_VENV=.venv
export UV_USE_PYPACKAGES=1
export MSYS_NO_PATHCONV=1
export PUPPETEER_EXECUTABLE_PATH="/usr/bin/chromium"
export PATH="/usr/local/typst/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# NVM
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Aliases
alias ls="ls --group-directories-first -a -h -p --format=vertical -s --color=always"
alias bat="batcat"
alias fcat='fzf --preview "batcat --style=numbers --color=always --line-range=:500 {}"'
alias zshconfig="nano ~/.zshrc"
alias cht="cht.sh"
ZSHRC_EOF
fi

# ── Typst PATH for vscode user ─────────────────────────────────────────────
if ! grep -q 'typst' /home/vscode/.bashrc 2>/dev/null; then
    echo 'export PATH="/usr/local/typst/bin:$PATH"' >> /home/vscode/.bashrc
fi

# ── Jupyter kernel registration ────────────────────────────────────────────
echo "==> [post-create] Registering Jupyter kernel"
python -m ipykernel install --user --name python3 --display-name "Python 3 (Codespace)" || true

echo "==> [post-create] Post-creation setup complete"
echo "    Launch JupyterLab: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
