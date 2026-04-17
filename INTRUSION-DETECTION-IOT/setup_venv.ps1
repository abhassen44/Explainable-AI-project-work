$ErrorActionPreference = 'Stop'
# Create virtual environment
python -m venv .venv
# Activate and upgrade pip, then install requirements
.\.venv\Scripts\Activate.ps1


Write-Host "Virtual environment created at .venv and packages installed."