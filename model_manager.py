import json
import os
import joblib
import torch
import functools

CONFIG_PATH = "models_config.json"

def load_registry():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)

MODEL_REGISTRY = load_registry()

def save_registry():
    with open(CONFIG_PATH, 'w') as f:
        json.dump(MODEL_REGISTRY, f, indent=2)

def register_model(name: str, path: str, framework: str, input_type: str):
    MODEL_REGISTRY[name] = {
        "path": path,
        "framework": framework,
        "input_type": input_type
    }
    save_registry()
    # Invalidate the cache for this model to ensure fresh load if it was overwritten
    load_model.cache_clear()

@functools.lru_cache(maxsize=3)
def load_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry.")

    config = MODEL_REGISTRY[name]
    local_path = config["path"]

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Model file not found at {local_path}")

    if config["framework"] == "sklearn":
        model = joblib.load(local_path)
    elif config["framework"] == "pytorch":
        model = torch.jit.load(local_path)
        model.eval()
    else:
        raise ValueError(f"Unsupported framework: {config['framework']}")

    print(f"Loaded model {name} into LRU cache.")
    return model

def predict(name: str, data):
    model = load_model(name)
    config = MODEL_REGISTRY[name]

    if config["framework"] == "sklearn":
        # sklearn models generally expect 2D arrays
        import numpy as np
        return model.predict([data]).tolist()
    elif config["framework"] == "pytorch":
        tensor = torch.tensor(data).float()
        with torch.no_grad():
            output = model(tensor)
            if hasattr(output, 'detach'):
                return output.detach().numpy().tolist()
            return output.tolist()
    else:
        raise ValueError("Unsupported framework")
