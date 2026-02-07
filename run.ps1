# Kinetic Analytics V2.0 - Launch Script (PowerShell)
# Run this script to start both backend and frontend

Write-Host "=============================================="
Write-Host "  KINETIC ANALYTICS V2.0 - Launcher"
Write-Host "=============================================="
Write-Host ""

$projectRoot = $PSScriptRoot

# Check if Python environment exists
$pythonPath = "c:\Users\vitta\OneDrive\Desktop\Python 3.10\mp_env\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "[ERROR] Python environment not found at: $pythonPath" -ForegroundColor Red
    exit 1
}

# Start Backend
Write-Host "[1/2] Starting Backend Server..." -ForegroundColor Cyan
$backendPath = Join-Path $projectRoot "backend"
Start-Process -FilePath $pythonPath -ArgumentList "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080" -WorkingDirectory $backendPath -NoNewWindow

Write-Host "      Backend starting at: http://localhost:8080" -ForegroundColor Green
Write-Host "      WebSocket endpoint:  ws://localhost:8080/ws" -ForegroundColor Green
Write-Host ""

# Wait a moment for backend to initialize
Start-Sleep -Seconds 3

# Check if npm is available
$npmPath = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmPath) {
    Write-Host "[WARNING] npm not found. Please install Node.js to run the frontend." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Backend is running. You can test with:"
    Write-Host "  - Health check: http://localhost:8000/"
    Write-Host "  - WebSocket: Use browser console or wscat"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the backend."
    
    # Keep script running
    while ($true) { Start-Sleep -Seconds 1 }
}

# Install frontend dependencies if needed
$frontendPath = Join-Path $projectRoot "frontend"
$nodeModulesPath = Join-Path $frontendPath "node_modules"

if (-not (Test-Path $nodeModulesPath)) {
    Write-Host "[2/2] Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location $frontendPath
    npm install
    Pop-Location
}

# Start Frontend
Write-Host "[2/2] Starting Frontend Dev Server..." -ForegroundColor Cyan
Push-Location $frontendPath
npm run dev
Pop-Location

Write-Host ""
Write-Host "=============================================="
Write-Host "  Both services started!"
Write-Host "  Frontend: http://localhost:5173"
Write-Host "  Backend:  http://localhost:8080"
Write-Host "=============================================="
