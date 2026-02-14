param(
    [string]$PythonExe = "C:\Users\Max\code\xray_fracture_benchmark_venv\Scripts\python.exe"
)

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Write-Host "[1/4] Upgrading pip/setuptools/wheel..."
& $PythonExe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade packaging tools." }

Write-Host "[2/4] Installing CUDA-enabled PyTorch (cu128)..."
& $PythonExe -m pip install -r requirements-cu128.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch packages." }

Write-Host "[3/4] Installing base requirements..."
& $PythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install base requirements." }

Write-Host "[4/4] Verifying torch/cuda..."
& $PythonExe scripts\verify_torch_cuda.py
if ($LASTEXITCODE -ne 0) { throw "Torch/CUDA verification failed." }

Write-Host "Environment setup complete."
