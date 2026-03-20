#!/bin/bash
# on-create.sh — runs during GitHub Codespaces PREBUILD
# Heavy installs go here so they are cached in the prebuild snapshot.
set -euo pipefail

echo "==> [on-create] Starting prebuild setup"

REPO_ROOT="${CODESPACE_VSCODE_FOLDER:-$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_ROOT"
echo "==> [on-create] Working in: $REPO_ROOT"

# ── Python virtual environment ─────────────────────────────────────────────
echo "==> [on-create] Creating virtual environment with uv"
uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ── Python dependencies ────────────────────────────────────────────────────
echo "==> [on-create] Installing Python dependencies"
# Try bulk install first; fall back to per-package so one bad name doesn't
# abort the entire install (e.g. lsp/json-lsp/yaml-lsp may not exist on PyPI).
if ! uv pip install --no-cache-dir -r .binder/requirements.txt; then
    echo "WARNING: bulk install failed – retrying package by package"
    failed=()
    while IFS= read -r pkg; do
        [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
        uv pip install --no-cache-dir "$pkg" || failed+=("$pkg")
    done < .binder/requirements.txt
    if [ ${#failed[@]} -gt 0 ]; then
        echo "WARNING: the following packages could not be installed:"
        printf '  - %s\n' "${failed[@]}"
    fi
fi

# Register the venv kernel so JupyterLab can find it
python -m ipykernel install --user --name iceberg-ml --display-name "Python 3 (Iceberg ML)"

echo "==> [on-create] Prebuild setup complete"
