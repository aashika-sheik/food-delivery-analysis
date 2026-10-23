@echo off
REM One-command launcher for Windows: double-click this file
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -q -r requirements.txt
streamlit run app.py
