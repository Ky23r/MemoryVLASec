param(
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $VenvPath) {
    $VenvPath = Join-Path $ProjectRoot ".venv-cpu"
}
$VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
$Python = Join-Path $VenvPath "Scripts\python.exe"
$PipTemp = Join-Path $HOME ".pip_tmp"
$env:TMPDIR = $PipTemp
New-Item -ItemType Directory -Force -Path $PipTemp | Out-Null

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Host "Creating project-local Windows CPU environment at '$VenvPath'..."
    $Created = $false

    # Prefer the python.exe already selected by PATH. Unlike the Windows `py`
    # launcher, this also works when Python was installed by Anaconda or another
    # distribution that did not register every interpreter with py.exe.
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $VersionOutput = & python -c `
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $Version = "$VersionOutput".Trim()
            if ($Version -in @("3.10", "3.11")) {
                Write-Host "Using Python $Version from PATH to create the local environment."
                & python -m venv $VenvPath
                $Created = $LASTEXITCODE -eq 0
            }
        }
    }

    # Fall back to registered launcher runtimes, probing before selecting one.
    if (-not $Created -and (Get-Command py -ErrorAction SilentlyContinue)) {
        foreach ($CandidateVersion in @("3.10", "3.11")) {
            $VersionFlag = "-$CandidateVersion"
            & py $VersionFlag -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Using Python $CandidateVersion from the Windows launcher."
                & py $VersionFlag -m venv $VenvPath
                $Created = $LASTEXITCODE -eq 0
                if ($Created) {
                    break
                }
            }
        }
    }

    if (-not $Created -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw (
            "Python 3.10 or 3.11 is required, but no compatible runtime was found. " +
            "Install Python 3.11 from python.org and rerun this script."
        )
    }
} else {
    Write-Host "Reusing project-local Windows CPU environment at '$VenvPath'."
}

& $Python -m pip install --upgrade `
    "pip==25.1.1" "setuptools==75.8.0" "wheel==0.45.1"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the pinned packaging tools."
}

# This is deliberately the CPU wheel index. The Linux setup independently
# selects its wheel index through PYTORCH_INDEX_URL in setup_env.sh.
& $Python -m pip install `
    "torch==2.7.1" "torchvision==0.22.1" `
    --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the CPU PyTorch wheels."
}

$CpuDependencies = @(
    "numpy==1.26.4",
    "draccus==0.8.0",
    "timm==0.9.10",
    "transformers==4.40.1",
    "huggingface-hub==0.29.3",
    "tokenizers==0.19.1",
    "einops==0.8.0",
    "accelerate==0.29.3",
    "sentencepiece==0.1.99",
    "tqdm==4.66.4",
    "Pillow==10.3.0",
    "packaging==24.0",
    "requests==2.32.3",
    "rich==13.7.1",
    "PyYAML==6.0.2"
)
& $Python -m pip install @CpuDependencies
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Windows CPU integration dependencies."
}

# Install repository import paths without the Linux-only full requirements.
& $Python -m pip install -e $ProjectRoot --no-deps
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install MemoryVLASec in editable mode."
}

& $Python -c `
    "import torch; import attacks.badvla; import defenses.amemguard; import models.core.vla.memory_vla; print(f'PyTorch {torch.__version__}; CUDA runtime: {torch.version.cuda}; CPU integration imports ready')"
if ($LASTEXITCODE -ne 0) {
    throw "The Windows CPU integration import check failed."
}

Write-Host "Windows CPU environment is ready at '$VenvPath'."
Write-Host "No Conda environment was created or activated."
Write-Host "Run: powershell -ExecutionPolicy Bypass -File .\scripts\integration_test_cpu.ps1"
