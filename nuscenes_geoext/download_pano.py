import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import cv2
import time
from typing import Tuple
from nuscenes_geoext.utils.utils import load_json, save_json, get_streetview_metadata
from nuscenes_geoext.utils.transform import get_poses, derive_latlon
from nuscenes_geoext.utils.stitch import stitch_pano


# ==============================
# Google StreetView API Configuration
# ==============================
GOOGLE_API_KEY = ""

MAX_API_CALLS = 10000000


def download_pano(dataroot: str,
                  version: str,
                  streetview_out_path: str,
                  pano_dict_path: str,
                  frame_dict_path: str,
                  unavailable_path: str,
                  google_api_key: str) -> None:
    """
    Main function: Obtain Google Street View images for each frame in NuScenes
    """
    # Load JSON
    pano_dict = load_json(pano_dict_path)
    frame_dict = load_json(frame_dict_path)
    unavailable_dict = load_json(unavailable_path)
    api_calls = 0  # Track API call count (metadata counts as one call, image download counts as multiple)

    # Initialize NuScenes
    nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)
    
    # First pass: Calculate total number of frames
    print('Calculating total number of frames...')
    total_frames = 0
    all_scene_data = []
    for scene in nusc.scene:
        scene_name = scene['name']
        scene_token = scene['token']
        location = nusc.get('log', scene['log_token'])['location']
        poses = get_poses(nusc, scene_token)
        coordinates = derive_latlon(location, poses)
        num_coords = len(coordinates)
        # print(f'Scene {scene_name}: {num_coords} frames')
        all_scene_data.append({
            'scene_name': scene_name,
            'scene_token': scene_token,
            'location': location,
            'coordinates': coordinates
        })
        total_frames += num_coords
    
    print(f'Total frames to process: {total_frames}')
    if total_frames == 0:
        print('Warning: No frames found! Check if get_poses and derive_latlon are working correctly.')
        return
    
    print('Start processing NuScenes frames and obtaining street view images...')
    sys.stderr.flush()
    sys.stdout.flush()
    
    # Create progress bar for all frames
    pbar = tqdm(total=total_frames, desc="Processing frames", position=0, leave=True, 
                file=sys.stdout, dynamic_ncols=True, miniters=1, maxinterval=0.1)
    
    processed_count = 0
    for scene_data in all_scene_data:
        scene_name = scene_data['scene_name']
        scene_token = scene_data['scene_token']
        location = scene_data['location']
        coordinates = scene_data['coordinates']
        
        for coord in coordinates:
            frame_id = coord['token']  # Use ego_pose token as frame_id
            
            if frame_id in frame_dict:
                pano_id = frame_dict[frame_id]['pano_id']
                if pano_id in pano_dict:
                    if frame_id not in pano_dict[pano_id]['tokens']:
                        pano_dict[pano_id]['tokens'].append(frame_id)
                        save_json(pano_dict_path, pano_dict)
                    processed_count += 1
                    pbar.write(f"[{processed_count}/{total_frames}] Frame {frame_id} has been processed, skipping")
                    pbar.update(1)
                    pbar.refresh()
                    continue
                else:
                    frame_dict.pop(frame_id)
                

            if frame_id in unavailable_dict:
                processed_count += 1
                pbar.write(f"[{processed_count}/{total_frames}] Frame {frame_id} is unavailable, skipping")
                pbar.update(1)
                pbar.refresh()
                continue
            
            if api_calls >= MAX_API_CALLS:
                pbar.write(f"Reached the maximum API call limit ({MAX_API_CALLS})")
                save_json(pano_dict_path, pano_dict)
                save_json(frame_dict_path, frame_dict)
                pbar.close()
                return
            
            lat = coord['latitude']
            lon = coord['longitude']

            # Query metadata
            meta = get_streetview_metadata(lat, lon, api_key=google_api_key)
            api_calls += 1
            if not meta:
                processed_count += 1
                pbar.write(f"[{processed_count}/{total_frames}] No available street view for frame {frame_id}")
                unavailable_dict[frame_id] = {
                    "lat": lat,
                    "lon": lon,
                    "pano_id": None
                }
                save_json(unavailable_path, unavailable_dict)
                pbar.update(1)
                pbar.refresh()
                continue
            
            pano_id = meta['pano_id']
            pano_lat = meta['location']['lat']
            pano_lon = meta['location']['lng']
            pano_date = meta.get('date', 'unknown')
            
            pano = None
            pano_path = None
            
            if pano_id not in pano_dict:
                # Not recorded: Download and stitch panorama
                pbar.write(f"First encounter of pano_id={pano_id}, downloading and stitching panorama...")
                pano, pano_path = stitch_pano(pano_lat, pano_lon, pano_id, streetview_out_path, google_api_key)
                api_calls += 6  # 6 tiles
                
                # Check if download failed
                if pano is None or pano_path is None:
                    pbar.write(f"Failed to download panorama {pano_id}, adding to unavailable list")
                    unavailable_dict[frame_id] = True
                    save_json(unavailable_path, unavailable_dict)
                    processed_count += 1
                    pbar.update(1)
                    pbar.refresh()
                    continue
                
                # Record in pano_dict
                pano_dict[pano_id] = {
                    "lat": pano_lat,
                    "lon": pano_lon,
                    "date": pano_date,
                    "pano_path": pano_path,
                    "tokens": [frame_id]
                }
            else:
                # Already recorded: Load existing panorama
                pbar.write(f"Reuse existing pano_id={pano_id}")
                pano_dict[pano_id]['tokens'].append(frame_id)
                
            
            # Record in frame_dict
            frame_dict[frame_id] = {
                "lat": lat,
                "lon": lon,
                "pano_id": pano_id
            }
            
            # Save JSON
            save_json(pano_dict_path, pano_dict)
            save_json(frame_dict_path, frame_dict)
            
            # Update progress bar
            processed_count += 1
            pbar.update(1)
            pbar.refresh()
            
            # Add delay to avoid API limits
            time.sleep(0.1)
    
    # Close progress bar
    pbar.close()
    print(f'\nProcessing completed! Total frames processed: {processed_count}/{total_frames}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Obtain Google Street View images for each frame in NuScenes')
    parser.add_argument('--dataroot', type=str, default='./nuScenes/',
                        help="Path to the NuScenes dataset")
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Dataset version')
    parser.add_argument('--api_key', type=str, default=GOOGLE_API_KEY,
                        help='Google API key')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Path to the output directory')
    args = parser.parse_args()

    if args.api_key is None:
        raise ValueError('Google API key is not set')

    streetview_out_path = os.path.join(args.output_dir, "streetview")
    pano_dict_path = os.path.join(args.output_dir, "pano_metadata.json") 
    frame_dict_path = os.path.join(args.output_dir, "frame_metadata.json")
    unavailable_path = os.path.join(args.output_dir, "unavailable_metadata.json")

    download_pano(args.dataroot,
                  args.version,
                  streetview_out_path=streetview_out_path,
                  pano_dict_path=pano_dict_path,
                  frame_dict_path=frame_dict_path,
                  unavailable_path=unavailable_path,
                  google_api_key=args.api_key)