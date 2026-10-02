@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8502
