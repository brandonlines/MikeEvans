#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH MIKE_MODEL
if [[ ! -x .venv/bin/python || ! -f face_landmarker.task ]]; then
 echo 'This folder needs setup. Run: bash 01_Setup.command'
 exit 1
fi
exec .venv/bin/python app.py --head-motion 0.7 --brow-strength 1.0 "$@"
