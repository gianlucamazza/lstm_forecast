#!/bin/bash

# Train the model
echo "Training the model..."
python src/train.py --config "config.json"

# Run prediction script
echo "Running prediction..."
python src/predict.py --ticker "BTC-USD" --model_path "model.pth" --config config.json
