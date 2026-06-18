$ErrorActionPreference = "Stop"

Write-Host "Starting Vietnamese Student Feedback NLP Demo..."
Write-Host ""

$ProjectRoot = (Get-Location).Path
Write-Host "Project root: $ProjectRoot"

python -m pip install -r requirements_demo.txt

Write-Host ""
Write-Host "Open browser:"
Write-Host "http://127.0.0.1:8000"
Write-Host ""

python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
