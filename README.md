# Cloud-Ready Multi-Tenant ML Inference Platform

**A lightweight, cloud-ready platform that allows dynamic upload, registration, and serving of machine learning models through a unified API with scalable, containerized architecture.**

## Problem Statement

Deploying and managing multiple ML models in production is often complex, requiring separate endpoints, infrastructure setup, and tight coupling between model logic and APIs. This creates friction in scaling, updating, and maintaining ML systems.

## Solution

This project introduces a registry-driven ML serving platform where models can be dynamically uploaded and served through a single generic API without modifying backend code. The system abstracts framework-specific logic and provides a consistent inference interface.

## Core Features

- **Dynamic model onboarding via API** (no redeployment needed)
- **Unified inference endpoint** for multiple ML frameworks
- **Support for Scikit-learn and PyTorch models**
- **Lazy loading with caching** to optimize performance
- **Batch inference support** for efficient processing
- **API key-based access control** for secure usage
- **Containerized deployment** for portability across environments

## System Design (Simplified)

Client requests are handled by a FastAPI service that routes inference calls based on a model registry. Models are loaded on demand and cached in memory. The system is containerized and can be extended with additional services like Redis for asynchronous processing.

## Cloud Relevance

The platform is designed with cloud principles in mind:
- **Containerized architecture** (Docker)
- **Service-oriented design** (API + optional queue)
- **Horizontal scalability** via multiple instances
- **Decoupled model management** using a registry
- **Deployment-ready** on platforms like Railway or Render

## Tech Stack

- **Backend:** FastAPI, Python
- **ML Support:** Scikit-learn, PyTorch, NumPy
- **Data Handling:** Joblib, Pydantic
- **Deployment:** Docker, Docker Compose
- **Frontend:** HTML, CSS, JavaScript

## What Makes It Strong

- Eliminates need for hardcoded ML endpoints
- Demonstrates real-world ML system design
- Balances simplicity with scalability
- Cloud-agnostic and easily deployable

## Future Scope

- Model versioning and rollback
- Async inference with queue-based processing
- Per-model monitoring and metrics
- Rate limiting and multi-tenant isolation

## Quick Start

```bash
# Activate virtual environment
source venv-ardupilot/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List all registered models |
| `/upload_model` | POST | Upload and register a new model |
| `/predict/{model_name}` | POST | Run inference on a model |

## Usage

### List Models
```bash
curl http://127.0.0.1:8000/models
```

### Upload a Model
```bash
curl -X POST "http://127.0.0.1:8000/upload_model" \
  -F "name=my_model" \
  -F "framework=sklearn" \
  -F "input_type=tabular" \
  -F "file=@models/fraud.pkl"
```

### Run Inference
```bash
curl -X POST "http://127.0.0.1:8000/predict/my_model" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[100.0, 50.0, 1]]}'
```

## Project Structure

```
.
├── main.py                 # FastAPI application
├── model_manager.py        # Model loading & caching
├── models_config.json      # Model registry
├── index.html              # Frontend dashboard
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── deploy.sh               # Deployment script
```

## License

MIT