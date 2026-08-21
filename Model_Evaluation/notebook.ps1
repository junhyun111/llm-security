param(
    [ValidateSet("lab", "train", "evaluation")]
    [string]$Notebook = "lab"
)

$ErrorActionPreference = "Stop"
$evaluationRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$env:UV_CACHE_DIR = Join-Path $evaluationRoot "cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $evaluationRoot "cache\python"
$env:PYTHONPATH = Join-Path $evaluationRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

$target = if ($Notebook -eq "lab") {
    $evaluationRoot
} else {
    Join-Path $evaluationRoot ($Notebook + ".ipynb")
}

& uv run --python 3.13 --no-project `
    --with-requirements (Join-Path $evaluationRoot "requirements.txt") `
    jupyter lab $target
exit $LASTEXITCODE
