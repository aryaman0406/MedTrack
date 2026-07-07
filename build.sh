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

# Run database migrations for PostgreSQL (alter password column)
echo "Running database migrations..."
python -c "
import os
from sqlalchemy import create_engine, text

database_url = os.getenv('DATABASE_URL', '')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

if database_url and 'postgresql' in database_url:
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Alter password column to VARCHAR(256) if it exists
            conn.execute(text('ALTER TABLE \"users\" ALTER COLUMN password TYPE VARCHAR(256)'))
            conn.commit()
            print('Password column migrated successfully')
    except Exception as e:
        print(f'Migration note: {e}')
else:
    print('Skipping PostgreSQL migration (not using PostgreSQL)')
" || echo "Migration completed or not needed"

echo "Build completed successfully!"
