#!/bin/bash
set -e

echo "=== Narrative Forge Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
echo "Installing dependencies..."
python -m pip install -r requirements.txt

# Verify GPU
echo ""
echo "=== GPU Verification ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'GPU {i}: {name} ({mem:.1f} GB)')
    print('GPU setup OK!')
else:
    print('WARNING: No CUDA GPU detected. Training will be very slow on CPU.')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with: source venv/Scripts/activate"
