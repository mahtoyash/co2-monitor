$ErrorActionPreference = "Stop"
$sourceDir = "c:\Users\HP\Downloads\co2-monitor-main\co2-monitor-main"
$targetDir = "c:\Users\HP\Downloads\co2-monitor-deploy"

Write-Host "Cleaning up old deploy directory if it exists..."
if (Test-Path $targetDir) { Remove-Item -Recurse -Force $targetDir }

Write-Host "Cloning repository..."
git clone https://github.com/mahtoyash/co2-monitor.git $targetDir

Write-Host "Navigating to repository..."
Set-Location $targetDir

Write-Host "Removing old TFJS model files..."
if (Test-Path "model\group*.bin") { Remove-Item -Force "model\group*.bin" }
if (Test-Path "model\model.json") { Remove-Item -Force "model\model.json" }

Write-Host "Copying updated application files..."
Copy-Item -Path "$sourceDir\model\base_weights.json" -Destination "model\base_weights.json" -Force
Copy-Item -Path "$sourceDir\js\app.js" -Destination "js\app.js" -Force

Write-Host "Copying training artifacts..."
Copy-Item -Path "$sourceDir\co2_lstm_classroom.py" -Destination "." -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$sourceDir\co2_lstm_classroom.pth" -Destination "." -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$sourceDir\feature_scaler_classroom.pkl" -Destination "." -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$sourceDir\target_scaler_classroom.pkl" -Destination "." -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$sourceDir\limits.json" -Destination "." -Force -ErrorAction SilentlyContinue

Write-Host "Committing and pushing to GitHub..."
git add .
git commit -m "feat: Migrate CO2 predictions to new PyTorch Classroom LSTM model"
git push origin main
Write-Host "Done! Changes have been pushed to GitHub."
