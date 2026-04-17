$ErrorActionPreference = 'Stop'
# Create virtual environment
python -m venv .venv
# Activate and upgrade pip, then install requirements
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Virtual environment created at .venv and packages installed."