#!/bin/bash

# Clear logs
echo "Clearing logs..."
rm -rf logs/*.log

# Clean up the data directory
echo "Cleaning up the data and reports directory..."
if [ "$1" == "--skip-training" ]; then
    echo "Skipping data cleanup..."
else
    rm -rf data/*.csv
    rm -rf reports/*.csv
fi

# Clean up the png and html directory
echo "Cleaning up the png directory..."
rm -rf png/*.png
echo "Cleaning up the html directory..."
rm -rf html/*.html


# Clean up the reports directory
echo "Cleaning up the reports directory..."
rm -rf reports/*.csv

# Train the model
# if argument is --skip-training, then skip training
if [ "$1" == "--skip-training" ]; then
    echo "Skipping training..."
else
    echo "Training the model..."
    python src/train.py --config config.json
fi

# Run prediction script
echo "Running prediction..."
python src/predict.py --config config.json

# Update HTML/docs for the prediction
echo "Updating HTML..."
python src/generate_html.py
