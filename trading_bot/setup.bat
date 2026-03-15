@echo off
setlocal

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

where python >nul 2>&1
if errorlevel 1 (
  echo Error: Python 3.11+ is required but was not found in PATH.
  exit /b 1
)

python -c "import sys; v=sys.version_info; print(f'Using Python {sys.version.split()[0]}'); sys.exit(0 if v >= (3,11) else 1)"
if errorlevel 1 (
  echo Error: Python 3.11+ is required.
  exit /b 1
)

if not exist venv (
  python -m venv venv
)

call venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist .env (
  copy .env.example .env
) else (
  echo .env already exists — skipping to protect your keys
)

if not exist models (
  mkdir models
)

echo Setup complete. Next step: open .env and add your API keys.
endlocal
