param(
    [ValidateSet("lab", "train", "evaluation")]
    [string]$Notebook = "lab"
)

$ErrorActionPreference = "Stop"
$evaluationRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$env:PYTHONPATH = Join-Path $evaluationRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

$target = if ($Notebook -eq "lab") {
    $evaluationRoot
} else {
    Join-Path $evaluationRoot ($Notebook + ".ipynb")
}

$venvCandidates = @(
    (Join-Path $evaluationRoot ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path $evaluationRoot -Parent) ".venv\Scripts\python.exe")
)
$venvPython = $venvCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $venvPython) {
    throw "가상환경이 없습니다. Model_Evaluation 또는 프로젝트 루트의 .venv에 ipykernel/jupyter를 설치하세요."
}
& $venvPython -m jupyter lab $target
exit $LASTEXITCODE
