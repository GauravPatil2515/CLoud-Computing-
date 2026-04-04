from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
import os
import shutil
import model_manager

app = FastAPI(title="Multi-Tenant Model Platform")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/upload_model")
def upload_model(
    name: str = Form(...),
    framework: str = Form(...),
    input_type: str = Form(...),
    file: UploadFile = File(...)
):
    """Dynamic Onboarding: Accept a model file and map it in the registry."""
    if framework not in ["sklearn", "pytorch"]:
        raise HTTPException(status_code=400, detail="Only 'sklearn' or 'pytorch' allowed.")
    
    os.makedirs("models", exist_ok=True)
    file_path = f"models/{file.filename}"
    
    # Save the file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Register the model details
    model_manager.register_model(name, file_path, framework, input_type)
    
    return {"status": "Model successfully onboarded", "name": name}

@app.post("/predict/{model_name}")
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
