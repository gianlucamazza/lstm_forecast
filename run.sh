#!/bin/bash

# Clear logs
echo "Clearing logs..."
rm -rf logs/*.log

# Clean up the data directory
echo "Cleaning up the data directory..."
rm -rf data/*.csv

# Clean up the model directory
echo "Cleaning up the model..."
rm -rf model.pth

# Train the model
echo "Training the model..."
python src/train.py --config config.json

# Run prediction script
echo "Running prediction..."
python src/predict.py --config config.json
