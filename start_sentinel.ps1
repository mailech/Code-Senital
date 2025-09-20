# Self-Healing Codebase Sentinel Startup Script
Write-Host "Starting Self-Healing Codebase Sentinel..." -ForegroundColor Green
Write-Host ""

Write-Host "Testing application components..." -ForegroundColor Yellow
python quick_test.py
Write-Host ""

Write-Host "Starting FastAPI server..." -ForegroundColor Yellow
Write-Host "Open http://localhost:8000/dashboard in your browser" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
