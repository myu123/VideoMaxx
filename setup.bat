@echo off
echo ===============================================
echo  VideoMaxx - Full Environment Setup
echo ===============================================
echo.
echo Prerequisites:
echo   - Python 3.11 installed (py -3.11 must work)
echo   - ffmpeg installed and on PATH
echo   - NVIDIA GPU driver installed (CUDA 13.1)
echo   - Node.js 18+ installed
echo.

cd /d "%~dp0"

REM --- Create venv with Python 3.11 ---
echo [1/5] Creating Python 3.11 virtual environment...
py -3.11 -m venv venv
if errorlevel 1 (
    echo ERROR: Python 3.11 not found.
    echo Install from https://www.python.org/downloads/release/python-3119/
    echo Make sure to check "Add to PATH" during install.
    pause
    exit /b 1
)
call venv\Scripts\activate

REM --- Install PyTorch with CUDA 13.0 (compatible with driver CUDA 13.1) ---
echo [2/5] Installing PyTorch with CUDA 13.0 support for RTX 5070 Ti...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

REM --- Install remaining Python deps ---
echo [3/5] Installing Python dependencies...
pip install -r backend\requirements.txt

REM --- Install Angular deps ---
echo [4/5] Installing Angular frontend dependencies...
cd frontend
call npm install
cd ..

REM --- Verify ---
echo [5/5] Verifying installation...
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
python -c "import cv2; print(f'OpenCV {cv2.__version__} OK')"
ffmpeg -version 2>nul | findstr "version" || echo WARNING: ffmpeg not found on PATH

echo.
echo ===============================================
echo  Setup complete!
echo  Run start_backend.bat and start_frontend.bat
echo ===============================================
pause
