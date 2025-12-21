import os

CACHE_DIR = "/runpod-volume/huggingface-cache/hub"

def find_model_path(model_name: str) -> str | None:
    """
    Find the path to a cached model on RunPod.
    
    Args:
        model_name: The model name from Hugging Face
        (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')
    
    Returns:
        The full path to the cached model, or None if not found
    """
    if not model_name:
        return None
        
    # Convert model name format: "Org/Model" -> "models--Org--Model"
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(CACHE_DIR, f"models--{cache_name}", "snapshots")
    
    # Check if the model exists in cache
    if os.path.exists(snapshots_dir):
        try:
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                # Return the path to the first (usually only) snapshot
                return os.path.join(snapshots_dir, snapshots[0])
        except OSError:
            return None
    
    return None
