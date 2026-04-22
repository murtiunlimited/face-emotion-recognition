#!/bin/bash

set -e  # stop on error

echo "Starting project setup..."

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists, we are skipping..."
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# 4. Run preprocessing
echo "Running preprocessing..."
python -m src.data.preprocess

# 5. Train model
echo "Training model..."
python -m src.models.train

# 6. Launch frontend in Chrome (macOS)
echo "Opening frontend in Chrome..."
open -a "Google Chrome" frontend/index.html

# 7. Start FastAPI server
echo "Starting API server..."
uvicorn api.app:app --reload
