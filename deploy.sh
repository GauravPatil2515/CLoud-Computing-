#!/bin/bash

# Cloud Run ML API Deployment Script
# This encapsulates the 'gcloud builds' and 'gcloud run deploy' logic.

# Ensure simple error checking
set -e

# ================= Configuration Variables =================
# Replace these with your actual IDs/Names prior to deploying
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="ml-prediction-service"
REGION="us-central1"

# The bucket storing model artifacts
MODEL_BUCKET_NAME="my-ml-models-bucket"

# ================= Deployment =================

echo "Deploying Cloud Run service '$SERVICE_NAME' to project '$PROJECT_ID' in region '$REGION'..."
echo "Models will be fetched dynamically from GCS Bucket: gs://$MODEL_BUCKET_NAME"

# Build and Push image while deploying directly from source
# gcloud builds submit happens implicitly here.
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10 \
  --concurrency 80 \
  --set-env-vars MODEL_BUCKET_NAME=$MODEL_BUCKET_NAME

echo ""
echo "=========================================="
echo "Deployment Pipeline Complete!"
echo "Check Cloud Run console for the URL endpoint."
echo "Use the '/health' endpoint to verify basic functionality."
