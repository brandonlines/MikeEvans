#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo "Setup stopped. Copy the error above; you can rerun this setup safely."' ERR
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Use native Terminal on an Apple Silicon Mac (not Rosetta).'; exit 1; }
export PATH="$PWD/.tools:/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_PYTHON_PREFERENCE=only-managed
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH MIKE_MODEL
if ! command -v uv >/dev/null 2>&1; then
 echo 'Installing uv from Astral into this folder (no Homebrew required).'
 mkdir -p .tools
 curl --fail --location --silent --show-error https://astral.sh/uv/install.sh -o .tools/install-uv.sh
 UV_UNMANAGED_INSTALL="$PWD/.tools" sh .tools/install-uv.sh
fi
echo 'Installing Python 3.12 and pinned packages into this folder...'
uv sync --frozen --no-dev --python 3.12
.venv/bin/python download_model.py
.venv/bin/python app.py --check
echo 'SETUP PASSED. Start with: bash 02_Start_Layered.command'
