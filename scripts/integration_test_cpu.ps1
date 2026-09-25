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

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw (
        "The local Windows CPU environment does not exist at '$VenvPath'. Create it first:`n" +
        "  powershell -ExecutionPolicy Bypass -File .\scripts\setup_env_cpu.ps1"
    )
}

Set-Location -LiteralPath $ProjectRoot

# Make accidental GPU use impossible. This launcher does not activate Conda.
$env:CUDA_VISIBLE_DEVICES = ""
$env:TOKENIZERS_PARALLELISM = "false"

& $Python -m unittest tests.test_cpu_integration -v
if ($LASTEXITCODE -ne 0) {
    throw "CPU integration test failed with exit code $LASTEXITCODE."
}
