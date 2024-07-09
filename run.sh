#!/bin/bash

# Clear logs
echo "Clearing logs..."
rm -rf logs/*.log

# Clean up the data directory
echo "Cleaning up the data directory..."
rm -rf data/*.csv

# Clean up the png directory
echo "Cleaning up the png directory..."
rm -rf png/*.png

# Train the model
# if skip_training is set to true in the config file, the training will be skipped
if [ "$SKIP_TRAINING" = "true" ]; then
    echo "Skipping training..."
else
    echo "Training the model..."
    python src/train.py --config config.json
fi

# Run prediction script
echo "Running prediction..."
python src/predict.py --config config.json
