# run_pipeline.ps1
# End-to-end pipeline runner for VQC-ZTI CESNET experiments (Windows PowerShell)

$ErrorActionPreference = "Stop"

# Step 0: Move to repo root (the folder containing this script)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "`n[pipeline] Repo root:" (Get-Location) -ForegroundColor Cyan

# Step 1: Activate venv if present (optional)
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  Write-Host "`n[pipeline] Activating venv..." -ForegroundColor Green
  . .\.venv\Scripts\Activate.ps1
} else {
  Write-Host "`n[pipeline] WARNING: .venv not found. Use your existing environment or create one:" -ForegroundColor Yellow
  Write-Host "          py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

# Step 2: Preparing python Environment and installing required dependencies
if (Test-Path ".\requirements.txt") {
  Write-Host "`n[pipeline] Preparing python Environment and installing required dependencies..." -ForegroundColor Green
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
} else {
  Write-Host "`n[pipeline] WARNING: requirements.txt not found. Skipping dependency installation." -ForegroundColor Yellow
}

# Step 3: Preprocessing (build NPZ)
Write-Host "`n[pipeline] Preprocess (build NPZ)..." -ForegroundColor Green
python -m src.preprocess_cesnet

# Step 4: Make splits (group/time)
Write-Host "`n[pipeline] Make splits (group/time)..." -ForegroundColor Green
python -m src.make_splits

# Step 5: Train QNN
Write-Host "`n[pipeline] Train QNN..." -ForegroundColor Green
python -m src.train_qnn

# Step 6: Evaluate QNN
Write-Host "`n[pipeline] Evaluate QNN..." -ForegroundColor Green
python -m src.eval_qnn --all-runs

# Step 7: Train + Evaluate baseline
Write-Host "`n[pipeline] Train + Evaluate baseline..." -ForegroundColor Green
python -m src.baseline_train_eval

# Step 8: Plot results
Write-Host "`n[pipeline] Plot results..." -ForegroundColor Green
python -m src.plot_results --mode aggregate

# Final message
Write-Host "`n[pipeline] Pipeline completed. See ./artifacts, ./data/processed and ./data/processed/splits" -ForegroundColor Cyan