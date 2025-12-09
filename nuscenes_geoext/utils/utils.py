import os
import json
import pickle
from typing import Dict
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import time

STREETVIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
GOOGLE_API_KEY = ""

def load_json(file_path: str) -> Dict:
    """Load JSON file, return empty dict if not exists"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_json(file_path: str, data: Dict):
    """Save JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def safe_pickle_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def safe_pickle_load(path):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def get_streetview_metadata(lat: float, lon: float, api_key: str = None, max_retries: int = 5, retry_delay: float = 3.0) -> Dict:
    """
    Fetch metadata using Google Street View Metadata API

    Args:
        lat: Latitude
        lon: Longitude
        api_key: Google API key
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)

    Returns:
        Metadata dictionary, or None if failed
    """

    params = {
        "location": f"{lat},{lon}",
        "key": api_key if api_key is not None else GOOGLE_API_KEY
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(STREETVIEW_METADATA_URL, params=params, timeout=30)
            if response.status_code == 200:
                meta = response.json()
                if meta.get("status") == "OK":
                    return meta
                else:
                    print(f"Metadata API returned status: {meta.get('status')}")
                    return None
            else:
                print(f"Metadata API error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
        except Exception as e:
            print(f"Exception while fetching metadata: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                print(f"Max retries ({max_retries}) reached, giving up")
    
    return None

def get_streetview_image(lat: float, lon: float, heading: float, pitch: float = 0, fov: float = 60, api_key: str = None, max_retries: int = 5, retry_delay: float = 3.0) -> np.ndarray:
    """
    Fetch a Street View image using Google Street View API (perspective projection)

    Args:
        lat: Latitude
        lon: Longitude
        heading: Heading angle (0-360 degrees)
        pitch: Pitch angle (-90 to 90 degrees)
        fov: Horizontal field of view
        api_key: Google API key
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)

    Returns:
        numpy BGR image, or None if failed
    """
    params = {
        "size": "640x640",
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": GOOGLE_API_KEY if api_key is None else api_key
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(STREETVIEW_API_URL, params=params, timeout=30)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                print(f"Street View API error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
        except Exception as e:
            print(f"Exception while fetching image: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                print(f"Max retries ({max_retries}) reached, giving up")
    
    return None