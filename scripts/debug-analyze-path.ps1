param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectPath,

    [string]$ProjectName = "",

    [ValidateRange(0.0, 1.0)]
    [double]$Sensitivity = 0.5,

    [string]$Model = "",

    [string]$ApiKey = "",

    [string]$BackendUrl = "http://localhost:5080",

    [switch]$Wait
)

$ErrorActionPreference = "Stop"

$body = @{
    projectPath = $ProjectPath
    sensitivity = $Sensitivity
}

if ($ProjectName) {
    $body.projectName = $ProjectName
}

if ($Model) {
    $body.model = $Model
}

if ($ApiKey) {
    $body.apiKey = $ApiKey
}

Write-Host "[debug] Sending local project path to ASP.NET backend..."
Write-Host "[debug] Project: $ProjectPath"
Write-Host "[debug] Sensitivity: $Sensitivity"

$job = Invoke-RestMethod `
    -Method Post `
    -Uri "$BackendUrl/api/dev/runtime/analyze-path" `
    -ContentType "application/json" `
    -Body ($body | ConvertTo-Json -Depth 5)

$job | Format-List

if (-not $Wait) {
    Write-Host ""
    Write-Host "[debug] Job created."
    Write-Host "[debug] Status:"
    Write-Host "  $BackendUrl$($job.statusEndpoint)"
    Write-Host "[debug] Analysis:"
    Write-Host "  $BackendUrl$($job.analysisEndpoint)"
    exit 0
}

Write-Host ""
Write-Host "[debug] Waiting for analysis to finish..."

while ($true) {
    Start-Sleep -Seconds 2

    $status = Invoke-RestMethod `
        -Method Get `
        -Uri "$BackendUrl$($job.statusEndpoint)"

    Write-Host ("[{0,3}%] {1} - {2}" -f `
        $status.progress, $status.status, $status.message)

    if ($status.status -eq "completed") {
        Write-Host ""
        Write-Host "[debug] Analysis completed."
        $analysis = Invoke-RestMethod `
            -Method Get `
            -Uri "$BackendUrl$($job.analysisEndpoint)"

        $analysis | ConvertTo-Json -Depth 20
        break
    }

    if ($status.status -eq "failed") {
        throw "Runtime analysis failed: $($status.error)"
    }
}
