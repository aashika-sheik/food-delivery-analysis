#!/usr/bin/env bash
# One-command launcher for Mac / Linux:  ./run_app.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
