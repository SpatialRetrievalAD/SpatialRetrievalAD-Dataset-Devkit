import os
import cv2
import numpy as np
from typing import Tuple
from tqdm import tqdm
from nuscenes_geoext.utils.utils import get_streetview_image
from nuscenes_geoext.utils.pano import map_image_to_pano


def stitch_pano(lat: float,
                lon: float,
                pano_id: str,
                streetview_out_path: str,
                google_api_key: str,) -> Tuple[np.ndarray, str]:
    """
    Download multiple tiles and stitch them into a 360-degree panorama, saving to a file.
    """
    pano_parts = []
    masks = []
    meta_info = None

    streetview_out_dir_name = os.path.basename(streetview_out_path)
    pano_relative_path = os.path.join(streetview_out_dir_name, "panos")

    pano_dir = os.path.join(streetview_out_path, "panos")
    tile_dir = os.path.join(streetview_out_path, "tiles")
    os.makedirs(pano_dir, exist_ok=True)
    os.makedirs(tile_dir, exist_ok=True)

    pano_path = os.path.join(pano_dir, f"{pano_id}.jpg")
    pano_relative_path = os.path.join(pano_relative_path, f"{pano_id}.jpg")
    tile_dir = os.path.join(tile_dir, f"{pano_id}")
    os.makedirs(tile_dir, exist_ok=True)


    def map_2d(yaw):
        tile_path = os.path.join(tile_dir, f"{yaw}.jpg")
        if os.path.exists(tile_path):
            tqdm.write(f"Using existing tile yaw={yaw} for pano_id={pano_id}...")
            tile = cv2.imread(tile_path)
        else:
            tqdm.write(f"Fetching tile yaw={yaw} for pano_id={pano_id}...")
            tile = get_streetview_image(lat, lon, heading=yaw, fov=60, api_key=google_api_key)
            if tile is None:
                tqdm.write(f"Failed to fetch tile at yaw={yaw}")
                return None
            cv2.imwrite(tile_path, tile)
        pano_part, mask, meta = map_image_to_pano(tile, fov_h_deg=60, yaw_deg=yaw, pano_width=3600)
        pano_parts.append(pano_part)
        masks.append(mask)
        meta_info = meta
        return pano_parts, masks, meta_info

    yaw_angles = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    for yaw in yaw_angles:
        result = map_2d(yaw)
        if result is None:
            tqdm.write(f"Failed to download panorama {pano_id}, skipping...")
            return None, None
        _, _, meta_info = result


    pano_h, pano_w = meta_info["pano_h"], meta_info["pano_w"]
    pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    mask_all = np.zeros((pano_h, pano_w), dtype=np.uint8)

    # Stitching
    for pano_part, mask in zip(pano_parts, masks):
        pano[mask.astype(bool)] = pano_part[mask.astype(bool)]
        mask_all |= mask

    # Save the panorama image
    pano = pano[pano_h//3:2*pano_h//3, :]
    cv2.imwrite(pano_path, pano)
    tqdm.write(f"Saved pano {pano_path}")

    return pano, pano_relative_path

