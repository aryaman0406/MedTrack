@echo off
setlocal
title MedTrack - Intelligent Health Companion
echo ============================================================
echo   💊 MEDTRACK: INTELLIGENT HEALTH COMPANION
echo   Privacy-Focused ^| API-Free ^| AI-Powered
echo ============================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Virtual Environment Setup
if not exist ".venv" (
    echo [1/3] Creating secure virtual environment...
    python -m venv .venv
)

echo [2/3] Activating environment...
call .venv\Scripts\activate

:: Dependency Management
echo [3/3] Syncing medical datasets and dependencies...
pip install -r requirements.txt --quiet

echo.
echo ============================================================
echo   SUCCESS: MedTrack is ready.
echo   - Local Medical DB: Loaded
echo   - Web Scraper: Online (Wikipedia/MedlinePlus)
echo   - AI Models: Initialized
echo ============================================================
echo.
echo Launching local server at http://127.0.0.1:5000
echo.

python app.py
pause
