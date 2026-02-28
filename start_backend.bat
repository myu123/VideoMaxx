@echo off
echo ===============================================
echo  VideoMaxx - Starting Backend (FastAPI)
echo ===============================================

cd /d "%~dp0"

if not exist "venv" (
    echo ERROR: Run setup.bat first to create the environment.
    pause
    exit /b 1
)

call venv\Scripts\activate

echo.
echo Checking CUDA...
python -c "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE - will use CPU\"}'); print()"
echo Starting server on http://127.0.0.1:8000
python run_backend.py
