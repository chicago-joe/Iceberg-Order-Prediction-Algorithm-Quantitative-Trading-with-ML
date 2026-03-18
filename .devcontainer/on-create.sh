#!/bin/bash
# on-create.sh — runs during GitHub Codespaces PREBUILD
# Heavy, slow installs go here so they are cached in the prebuild snapshot.
set -euo pipefail

echo "==> [on-create] Starting prebuild setup"

REPO_ROOT="${CODESPACE_VSCODE_FOLDER:-/workspaces/$(basename "$(pwd)")}"
cd "$REPO_ROOT"

# ── Python dependencies ────────────────────────────────────────────────────
echo "==> [on-create] Installing Python dependencies with uv"
uv pip install --system --no-cache-dir -r .binder/requirements.txt

# ── Node / npm dependencies ────────────────────────────────────────────────
echo "==> [on-create] Installing npm packages"
npm install --force

# ── JupyterLab build (expensive, cache in prebuild) ───────────────────────
echo "==> [on-create] Building JupyterLab"
uv run --no-cache jupyter lab build --dev-build=False || echo "WARNING: jupyter lab build failed – continuing"

# ── Optional component build ───────────────────────────────────────────────
if [ -f "build-components.mjs" ]; then
    echo "==> [on-create] Building components"
    node build-components.mjs || echo "WARNING: build-components.mjs failed – continuing"
fi

echo "==> [on-create] Prebuild setup complete"
