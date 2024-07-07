#!/bin/bash

# Train the model
echo "Training the model..."
python train.py --config config.json

# Run prediction script
echo "Running prediction..."
python predict.py --ticker "BTC-USD" --model_path "model.pth"
