# Multi-Tenant Model Hosting Platform

A scalable, Hugging Face-like ML model hosting service built with FastAPI. Supports dynamic model onboarding, LRU caching, and standardized inference contracts.

## Features

- **Dynamic Model Onboarding** - Upload sklearn/PyTorch models via REST API
- **LRU Caching** - Memory-efficient model loading with `@functools.lru_cache(maxsize=3)`
- **Standardized Contracts** - Generic `/predict/{model_name}` endpoint for all models
- **Multi-tenant Ready** - Each model is isolated and independently scalable

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