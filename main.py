from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
import os
import shutil
import model_manager

app = FastAPI(title="Multi-Tenant Model Platform")

# Configure CORS (Restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Simple API Key Security
API_KEY = os.getenv("PLATFORM_API_KEY", "super-secret-default-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

class GenericInferenceRequest(BaseModel):
    # E.g., [[amount, time, loca]], or [[[[pixels]]]]
    inputs: List[Any]

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Multi-Tenant Platform Operational"}

@app.get("/models")
def list_models():
    """Return all dynamically registered models."""
    return {"models": model_manager.MODEL_REGISTRY}

@app.post("/upload_model", dependencies=[Depends(verify_api_key)])
def upload_model(
    name: str = Form(...),
    framework: str = Form(...),
    input_type: str = Form(...),
    file: UploadFile = File(...)
):
    """Dynamic Onboarding: Accept a model file and map it in the registry."""
    if framework not in ["sklearn", "pytorch"]:
        raise HTTPException(status_code=400, detail="Only 'sklearn' or 'pytorch' allowed.")
    
    # Optional: Check file extension
    valid_extensions = (".pkl", ".pt", ".pth", ".joblib")
    if not file.filename.endswith(valid_extensions):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Must be one of {valid_extensions}")
    
    os.makedirs("models", exist_ok=True)
    file_path = f"models/{file.filename}"
    
    # Save the file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Register the model details
    model_manager.register_model(name, file_path, framework, input_type)
    
    return {"status": "Model successfully onboarded", "name": name}

@app.post("/predict/{model_name}", dependencies=[Depends(verify_api_key)])
def predict_generic(model_name: str, payload: GenericInferenceRequest):
    """Standardized Inference Contract for all tenant models."""
    if model_name not in model_manager.MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found in registry.")
    
    try:
        data = payload.inputs
        
        # If the input type implies flat tabular but we sent array of arrays, or similar, 
        # the model_manager handles raw list data ingestion generically.
        # But we pass the first row for single inference or handle it inside predict.
        
        # We'll pass the exact payload list over to the manager which uses lru_cache loading
        result = model_manager.predict(model_name, data)
        return {"model": model_name, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
