#!/bin/bash

usage() {
  echo "Usage: $0 [--skip-training] [--rebuild-features]"
  exit 1
}

SKIP_TRAINING=false
REBUILD_FEATURES=false

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --skip-training) SKIP_TRAINING=true ;;
    --rebuild-features) REBUILD_FEATURES=true ;;
    *) usage ;;
  esac
  shift
done

# Clear logs
echo "Clearing logs..."
rm -rf logs/*.log

# Clean up the data and reports directories
echo "Cleaning up the data and reports directories..."
if [ "$SKIP_TRAINING" = true ]; then
  echo "Skipping data cleanup..."
else
  rm -rf data/*.csv
  rm -rf reports/*.csv
fi

# Clean up the png directory
echo "Cleaning up the png directory..."
rm -rf png/*.png

# Clean up the html directory
echo "Cleaning up the html directory..."
rm -rf html/*.html

# Clean up the reports directory
echo "Cleaning up the reports directory..."
rm -rf reports/*.csv

# Train the model
if [ "$SKIP_TRAINING" = true ]; then
  echo "Skipping training..."
elif [ "$REBUILD_FEATURES" = true ]; then
  echo "Rebuilding features..."
  python src/train.py --config config.json --rebuild-features
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
