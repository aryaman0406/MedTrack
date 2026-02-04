#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install system dependencies for OCR and PDF generation
apt-get update && apt-get install -y \
    tesseract-ocr \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
