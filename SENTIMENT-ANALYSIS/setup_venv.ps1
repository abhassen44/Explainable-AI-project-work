# ============================================================
# Sentiment Analysis — Virtual Environment Setup Script
# ============================================================

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Sentiment Analysis — Environment Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# 1. Create virtual environment
if (!(Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment already exists." -ForegroundColor Green
}

# 2. Activate virtual environment
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# 3. Install dependencies
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download NLTK data
Write-Host "[4/4] Downloading NLTK data..." -ForegroundColor Yellow
python -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('NLTK data downloaded successfully.')
"

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment manually, run:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To start training:" -ForegroundColor Cyan
Write-Host "  python train_svm.py" -ForegroundColor White
Write-Host "  python train_cnn_lstm.py" -ForegroundColor White
Write-Host ""
