#!/bin/bash

echo "🐳 Building pipeline images (MLflow ready)..."

STEPS=(
  "01_data_import"
  "02_data_clean"
  "03_merge"
  "04_encodage"
  "05_data_transformation"
  "06_resampling"
  "07_model_trainer"
  "08_model_evaluation"
)

for step in "${STEPS[@]}"; do
  echo ""
  echo "=============================="
  echo "🚀 Building pipeline-${step}"
  echo "=============================="

  docker build \
    -t "pipeline-${step}:latest" \
    -f "src/pipeline/${step}/Dockerfile" \
    .

  if [ $? -ne 0 ]; then
    echo "❌ Build failed for ${step}"
    exit 1
  fi

done

echo ""
echo "🔥 ALL PIPELINE IMAGES BUILT SUCCESSFULLY"
