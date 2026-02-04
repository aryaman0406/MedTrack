#!/usr/bin/env bash
# Exit on error
set -o errexit

# Upgrade pip first
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Try to install system dependencies for OCR and PDF generation (optional)
# These may fail on some platforms, which is OK - the app handles missing dependencies gracefully
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        tesseract-ocr \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf2.0-0 \
        libffi-dev \
        shared-mime-info || echo "Some system dependencies could not be installed (non-fatal)"
fi

echo "Build completed successfully!"
