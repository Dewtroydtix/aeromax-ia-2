@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo No se pudieron instalar dependencias. Comprueba Python e Internet.
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
