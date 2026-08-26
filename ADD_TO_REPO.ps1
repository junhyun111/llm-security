param(
    [Parameter(Mandatory=$true)]
    [string]$RepoPath
)

$ErrorActionPreference = "Stop"

$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = (Resolve-Path $RepoPath).Path

Write-Host "Target repo: $Repo" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    throw "지정한 경로가 Git 저장소 루트가 아닙니다: $Repo"
}

Push-Location $Repo
try {
    $branch = git branch --show-current
    Write-Host "Current branch: $branch" -ForegroundColor Yellow
    if ($branch -eq "main") {
        throw "main 브랜치입니다. feature/web-platform 브랜치로 이동한 뒤 다시 실행하세요."
    }
}
finally {
    Pop-Location
}

$Items = @(
    "backend",
    "frontend",
    "compose.platform.yml",
    "PLATFORM_README.md"
)

foreach ($item in $Items) {
    $src = Join-Path $Source $item
    $dst = Join-Path $Repo $item

    if (Test-Path $dst) {
        Write-Host "이미 존재하여 건너뜀: $item" -ForegroundColor Yellow
        continue
    }

    Copy-Item $src $dst -Recurse
    Write-Host "추가됨: $item" -ForegroundColor Green
}

Write-Host ""
Write-Host "완료. 다음 명령을 저장소 루트에서 실행하세요:" -ForegroundColor Green
Write-Host "docker compose -f compose.platform.yml up --build"
