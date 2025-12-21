# RunPod Cached Models Integration

This project is optimized for RunPod Serverless deployments. A key feature implemented is the ability to leverage **RunPod Cached Models** to significantly reduce cold start times and bandwidth costs.

## Key Information for Agents

- **Documentation**: Details on how the feature works are in [cache_models_runpod.md](file:///InfiniteTalk/assets/cache_models_runpod.md).
- **Utility Function**: A helper function `find_model_path(model_name)` is available in `src/app/runpod_utils.py`.
- **Usage**: When deploying workers that require large models (e.g., video generation, LLM), always check if a cached version can be used instead of downloading from Hugging Face every time.

### How to use the utility:
```python
from app.runpod_utils import find_model_path

model_path = find_model_path("Qwen/Qwen2.5-0.5B-Instruct")
if model_path:
    # Use the cached model path for inference
    print(f"Loading model from cache: {model_path}")
else:
    # Fallback to downloading or error handling
    print("Model not in cache.")
```

## Maintenance
If the RunPod cache directory structure changes, update the `CACHE_DIR` constant in `src/app/runpod_utils.py`.
