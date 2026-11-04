#!/usr/bin/env bash
# Runs the whole project. Each step prints progress as it goes.
set -euo pipefail
cd "$(dirname "$0")"
echo "[1/3] training models and running the fairness audit ..."; python src/predict_and_audit.py
echo "[2/3] estimating program effects (about 45 seconds) ...";   python src/program_effect.py
echo "[3/3] building dashboard and white paper ...";             python src/build_report.py
echo "done. see reports/ and README.md"
