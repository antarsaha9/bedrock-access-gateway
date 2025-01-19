import json
import hashlib
from pathlib import Path
from api.setting import CACHE_PATH

CACHE_DIR = Path(CACHE_PATH)
CACHE_DIR.mkdir(exist_ok=True)
def generate_cache_key(data: dict) -> str:
    """
    Generate a unique cache key based on input data.
    """
    json_data = json.dumps(data, sort_keys=True)  # Serialize the data
    return hashlib.sha256(json_data.encode()).hexdigest()  # Create a hash

def read_from_cache(key: str):
    """
    Read cached result from a file if it exists.
    """
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)['result']
    return None

def write_to_cache(key: str, query:dict, result: dict):
    """
    Write the result to a cache file.
    """
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(dict(query=query, result=result), f)