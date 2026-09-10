@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"

set VENV=venv
set PY=%VENV%\Scripts\python.exe
set MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
set HF_ENDPOINT=https://hf-mirror.com

echo ============================================
echo   RAG Knowledge Base - Launcher
echo ============================================
echo.

rem ---------- [1/4] Create venv ----------
if not exist "%PY%" (
  echo [1/4] First run: creating virtual environment...
  where py >nul 2>nul
  if %errorlevel%==0 (
    py -3.13 -m venv "%VENV%" 2>nul
    if not exist "%PY%" py -3 -m venv "%VENV%"
  ) else (
    python -m venv "%VENV%"
  )
  if not exist "%PY%" (
    echo [ERROR] Failed to create venv. Please install Python 3.10+ and retry.
    pause
    exit /b 1
  )
  echo [1/4] venv created.
) else (
  echo [1/4] venv already exists, skip.
)

rem ---------- [2/4] Install dependencies ----------
"%PY%" -c "import fastapi, chromadb, sentence_transformers" >nul 2>nul
if %errorlevel% neq 0 (
  echo [2/4] First run: installing dependencies, please wait for progress bars...
  echo.
  echo   -- Upgrade pip --
  "%PY%" -m pip install -U pip
  echo.
  echo   -- Install torch (CPU version, ~190MB) --
  "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  if %errorlevel% neq 0 (
    echo   [NOTE] Official torch source failed, retry with Tsinghua mirror...
    "%PY%" -m pip install torch -i %MIRROR%
  )
  echo.
  echo   -- Install other dependencies (Tsinghua mirror) --
  "%PY%" -m pip install -r backend\requirements.txt -i %MIRROR%
  echo.
  echo [2/4] Dependencies installed!
) else (
  echo [2/4] Dependencies already installed, skip.
)

rem ---------- [3/4] Start server ----------
echo [3/4] Starting server at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
"%PY%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

pause
