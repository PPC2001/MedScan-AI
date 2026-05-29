$ErrorActionPreference = "Stop"

Write-Host "==================================="
Write-Host "  Starting MedScan AI Backend"
Write-Host "==================================="

# Make sure we are in the script's directory
Set-Location -Path $PSScriptRoot

Write-Host "1. Starting infrastructure (Docker)..." -ForegroundColor Cyan
docker compose up -d

Write-Host "2. Installing/Syncing dependencies..." -ForegroundColor Cyan
uv sync

Write-Host "3. Starting Celery worker in background..." -ForegroundColor Cyan
# Start Celery in a separate background job
$celeryJob = Start-Job -ScriptBlock {
    Set-Location -Path $using:PSScriptRoot
    uv run python -m celery -A medscan.tasks.celery_app.celery_app worker --loglevel=info -Q pipeline --concurrency=2
}

Write-Host "4. Starting API Server..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop both the API and the Celery worker." -ForegroundColor Yellow

try {
    # Start API in the foreground
    uv run python -m uvicorn medscan.api.main:app --reload --port 8000 --host 0.0.0.0
}
finally {
    Write-Host "`nShutting down services..." -ForegroundColor Cyan
    Stop-Job -Job $celeryJob
    Remove-Job -Job $celeryJob -Force
    Write-Host "Services stopped." -ForegroundColor Green
}
