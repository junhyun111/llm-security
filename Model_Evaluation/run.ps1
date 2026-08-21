param(
    [ValidateSet("index", "build-pilot", "evaluate-analyzer", "run-initial")]
    [string]$Command = "run-initial",
    [string]$Config = "",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$evaluationRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if (-not $Config) {
    $Config = Join-Path $evaluationRoot "configs\pilot.toml"
}
$env:UV_CACHE_DIR = Join-Path $evaluationRoot "cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $evaluationRoot "cache\python"
$env:PYTHONPATH = Join-Path $evaluationRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

$arguments = @(
    "run",
    "--python", "3.13",
    "--no-project",
    "--with-requirements", (Join-Path $evaluationRoot "requirements.txt"),
    "python", "-m", "model_evaluation",
    $Command,
    "--config", $Config
)
if ($Rebuild -and $Command -in @("index", "run-initial")) {
    $arguments += "--rebuild"
}

& uv @arguments
exit $LASTEXITCODE

