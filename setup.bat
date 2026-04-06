@echo off
echo === Narrative Forge Setup ===

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

pip install --upgrade pip

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo === GPU Verification ===
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('WARNING: No GPU detected')"

echo.
echo === Setup Complete ===
echo Activate with: venv\Scripts\activate.bat
