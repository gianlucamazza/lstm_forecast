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
  TRAIN_SUCCESS=true
elif [ "$REBUILD_FEATURES" = true ]; then
  echo "Rebuilding features..."
  python src/train.py --config config.json --rebuild-features
  if [ $? -eq 0 ]; then
    TRAIN_SUCCESS=true
  else
    TRAIN_SUCCESS=false
  fi
else
  echo "Training the model..."
  python src/train.py --config config.json
  if [ $? -eq 0 ]; then
    TRAIN_SUCCESS=true
  else
    TRAIN_SUCCESS=false
  fi
fi

# Run prediction script only if training was successful
if [ "$TRAIN_SUCCESS" = true ]; then
  echo "Running prediction..."
  python src/predict.py --config config.json
  if [ $? -eq 0 ]; then
    PREDICT_SUCCESS=true
  else
    PREDICT_SUCCESS=false
  fi
else
  echo "Training failed, skipping prediction and HTML update."
  PREDICT_SUCCESS=false
fi

# Update HTML/docs for the prediction only if prediction was successful
if [ "$PREDICT_SUCCESS" = true ]; then
  echo "Updating HTML..."
  python src/generate_html.py
else
  echo "Prediction failed, skipping HTML update."
fi
