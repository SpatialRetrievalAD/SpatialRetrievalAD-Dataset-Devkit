import os
import io
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial
from multiprocessing import get_context, cpu_count
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes_geoext.utils.utils import load_json, save_json, safe_pickle_dump, safe_pickle_load
from nuscenes_geoext.utils.transform import get_camera_yaw_pitch
from nuscenes_geoext.utils.pano import extract_view_from_pano
from nuscenes_geoext.utils.transform import get_poses, derive_latlon, gps_ranges, get_xy_by_latlon, get_pano_intrinsic
from nuscenes_geoext.utils.sat import find_key_for_location, get_rectangle_corners, crop_quadrilateral, get_satellite_homography


# Multiprocessing global variables (injected by initializer)
g_nusc = None
g_geo_root = None
g_streetview_size = None
g_sate_size = None
g_streetview_fov = None
g_frame = None
g_pano = None
g_unavail = None
g_pano_img_cache = {}
g_gps_ranges = None


class NuScenesGeoExt:
    def __init__(self, 
                 dataroot: str, 
                 version: str, 
                 geoext_dataroot: str,
                 streetview_fov=59, 
                 streetview_size=(512, 512),
                 sate_size=(400, 400),
                 cache_flag: bool = False,
                 max_workers: int = None,
                 nusc: NuScenes = None):
        self.dataroot = dataroot
        self.version = version
        self.geoext_dataroot = geoext_dataroot
        self.pano_dict_path = os.path.join(geoext_dataroot, "pano_metadata.json")
        self.frame_dict_path = os.path.join(geoext_dataroot, "frame_metadata.json")
        self.unavailable_path = os.path.join(geoext_dataroot, "unavailable_metadata.json")
        self.streetview_fov = streetview_fov
        self.streetview_size = streetview_size
        self.sate_size = sate_size
        self.cache_flag = cache_flag
        
        # Set max_workers: if None, use max(1, cpu_count() - 2) to leave cores for system
        if max_workers is None:
            total_cores = cpu_count()
            self.max_workers = max(1, total_cores - 2)  # Reserve 2 cores for 
            if self.max_workers > 32:
                self.max_workers = 32
        else:
            self.max_workers = max(1, min(max_workers, cpu_count()))  # Clamp to [1, cpu_count()]

        self.pano_dict = load_json(self.pano_dict_path)
        self.frame_dict = load_json(self.frame_dict_path)
        self.unavailable_dict = load_json(self.unavailable_path)

        if nusc is not None:
            self.nusc = nusc
        else:
            self.nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)

        streetview_data_name = f"streetview_data_{version}.pkl"
        satellite_data_name = f"satellite_data_{version}.pkl"
        self.streetview_data_path = os.path.join(geoext_dataroot, streetview_data_name)
        self.satellite_data_path = os.path.join(geoext_dataroot, satellite_data_name)

        # Load or generate streetview data
        p = safe_pickle_load(self.streetview_data_path)
        if p is None:
            self.generate_streetview_data()
        else:
            self.streetview_data = p

        # Load or generate satellite data
        p = safe_pickle_load(self.satellite_data_path)
        if p is None:
            self.generate_satellite_data()
        else:
            self.satellite_data = p

    def generate_streetview_data(self):
        streetview_data = {}
        pano_views_out_path = os.path.join(self.geoext_dataroot, "streetview/cams")
        os.makedirs(pano_views_out_path, exist_ok=True)

        scene_names = [s['name'] for s in self.nusc.scene]
        total_scenes = len(scene_names)
        
        # Dynamic chunk size and worker adjustment based on dataset size
        # Ensure at least (max_workers * 2) tasks for load balancing
        NPROC = self.max_workers
        min_tasks = NPROC * 2  # Each worker should handle at least 2 tasks
        
        if total_scenes <= 10:  # mini dataset (10 scenes)
            CHUNK = max(1, total_scenes // min_tasks)
            NPROC = min(NPROC, max(1, total_scenes // 2))  # Reduce workers for tiny datasets
        elif total_scenes <= 100:  # Small dataset
            CHUNK = max(2, total_scenes // min_tasks)
        else:  # Large dataset (trainval: ~850 scenes, test: ~150 scenes)
            CHUNK = min(16, max(8, total_scenes // (NPROC * 4)))  # 4 tasks per worker
        
        chunks = [scene_names[i:i+CHUNK] for i in range(0, len(scene_names), CHUNK)]
        
        ctx = get_context("spawn")
        func = partial(NuScenesGeoExt._process_scene_streetview, pano_views_out_path=pano_views_out_path)
        
        print(f"Dataset: {total_scenes} scenes, using {NPROC} workers, {len(chunks)} tasks (chunk_size={CHUNK})")
        print(f"Starting streetview generation...")

        with ctx.Pool(
            processes=NPROC,
            initializer=NuScenesGeoExt._mp_init_streetview,
            initargs=(
                self.nusc.dataroot,
                self.nusc.version,
                self.geoext_dataroot,
                self.streetview_size,
                self.streetview_fov,
                self.frame_dict_path,
                self.pano_dict_path,
                self.unavailable_path
            )
        ) as pool:
            for part in tqdm(pool.imap_unordered(func, chunks),
                             total=len(chunks), desc="streetview-mp", dynamic_ncols=True):
                # Main process writes to disk and loads into memory
                for cam_token, data_dict in part.items():
                    # Create sample_token subfolder
                    sample_token = data_dict["sample_token"]
                    sample_folder = os.path.join(pano_views_out_path, sample_token)
                    os.makedirs(sample_folder, exist_ok=True)
                    
                    # Write JPEG to disk in sample_token folder
                    out_path = os.path.join(sample_folder, f"{cam_token}.jpg")
                    with open(out_path, "wb") as fw:
                        fw.write(data_dict["img_bytes"])
                    
                    # Store based on cache_flag
                    if self.cache_flag:
                        # Cache full image in memory
                        img = Image.open(io.BytesIO(data_dict["img_bytes"])).convert("RGB")
                        streetview_data[cam_token] = {
                            "streetview_img": img,
                            "streetview_path": out_path,
                            "streetview_extrinsic": data_dict["extrinsic"],
                            "streetview_intrinsic": data_dict["intrinsic"]
                        }
                    else:
                        # Only store relative path
                        rel_path = os.path.relpath(out_path, self.geoext_dataroot)
                        streetview_data[cam_token] = {
                            "streetview_img": rel_path,
                            "streetview_path": rel_path,
                            "streetview_extrinsic": data_dict["extrinsic"],
                            "streetview_intrinsic": data_dict["intrinsic"]
                        }

        safe_pickle_dump(streetview_data, self.streetview_data_path)
        self.streetview_data = streetview_data


    def generate_satellite_data(self):
        satellite_data = {}
        sat_orin_path = os.path.join(self.geoext_dataroot, "sat")
        sat_out_path = os.path.join(self.geoext_dataroot, "sat_slice")
        os.makedirs(sat_out_path, exist_ok=True)

        # Check satellite images availability (just for logging)
        for key in gps_ranges.keys():
            image_filename = os.path.join(sat_orin_path, f"{key}.png")
            if os.path.exists(image_filename):
                try:
                    with Image.open(image_filename) as img:
                        print(f"{image_filename}: Height = {img.height}, Width = {img.width}")
                except FileNotFoundError:
                    print(f"File {image_filename} not found.")
                except Exception as e:
                    print(f"An error occurred while opening {image_filename}: {e}")

        scene_names = [s['name'] for s in self.nusc.scene]
        total_scenes = len(scene_names)
        
        # Dynamic chunk size and worker adjustment based on dataset size
        # Ensure at least (max_workers * 2) tasks for load balancing
        NPROC = self.max_workers
        min_tasks = NPROC * 2  # Each worker should handle at least 2 tasks
        
        if total_scenes <= 10:  # mini dataset (10 scenes)
            CHUNK = max(1, total_scenes // min_tasks)
            NPROC = min(NPROC, max(1, total_scenes // 2))  # Reduce workers for tiny datasets
        elif total_scenes <= 100:  # Small dataset
            CHUNK = max(2, total_scenes // min_tasks)
        else:  # Large dataset (trainval: ~850 scenes, test: ~150 scenes)
            CHUNK = min(32, max(8, total_scenes // (NPROC * 4)))  # 4 tasks per worker
        
        chunks = [scene_names[i:i+CHUNK] for i in range(0, len(scene_names), CHUNK)]
        
        ctx = get_context("spawn")
        func = partial(NuScenesGeoExt._process_scene_satellite, sat_out_path=sat_out_path)

        print(f"Dataset: {total_scenes} scenes, using {NPROC} workers, {len(chunks)} tasks (chunk_size={CHUNK})")
        print(f'Starting satellite generation...')
        with ctx.Pool(
            processes=NPROC,
            initializer=NuScenesGeoExt._mp_init_satellite,
            initargs=(
                self.nusc.dataroot,
                self.nusc.version,
                self.geoext_dataroot,
                self.sate_size,
                sat_orin_path
            )
        ) as pool:
            for part in tqdm(pool.imap_unordered(func, chunks),
                             total=len(chunks), desc="satellite-mp", dynamic_ncols=True):
                # Main process writes to disk and loads into memory
                for lidar_sd_token, data_dict in part.items():
                    # Get sample_token for disk filename
                    try:
                        sample_data = self.nusc.get('sample_data', lidar_sd_token)
                        sample_token = sample_data['sample_token']
                        
                        # Write PNG bytes directly to disk (already encoded in subprocess)
                        sat_save_path = os.path.join(sat_out_path, f"{sample_token}.png")
                        with open(sat_save_path, "wb") as fw:
                            fw.write(data_dict["img_bytes"])
                        
                        # Store based on cache_flag
                        if self.cache_flag:
                            # Cache full image in memory (decode only once)
                            img = Image.open(io.BytesIO(data_dict["img_bytes"])).convert("RGB")
                            satellite_data[lidar_sd_token] = {
                                "satellite_data": img,
                                "satellite_path": sat_save_path,
                                "satellite_pix2ego": data_dict["H"]
                            }
                        else:
                            # Only store relative path
                            rel_path = os.path.relpath(sat_save_path, self.geoext_dataroot)
                            satellite_data[lidar_sd_token] = {
                                "satellite_data": rel_path,
                                "satellite_path": rel_path,
                                "satellite_pix2ego": data_dict["H"]
                            }
                    except Exception as e:
                        print(f"Error processing {lidar_sd_token}: {e}")
                        continue

        safe_pickle_dump(satellite_data, self.satellite_data_path)
        self.satellite_data = satellite_data


    @staticmethod
    def _mp_init_streetview(dataroot, version, geoext_root, streetview_size, streetview_fov,
                           frame_dict_path, pano_dict_path, unavailable_path):
        """Initialize worker process once to avoid repeatedly building NuScenes."""
        global g_nusc, g_geo_root, g_streetview_size, g_streetview_fov, g_frame, g_pano, g_unavail
        g_nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)
        g_geo_root = geoext_root
        g_streetview_size = streetview_size
        g_streetview_fov = streetview_fov
        # Load dictionaries in subprocess instead of passing via pickle
        g_frame = load_json(frame_dict_path)
        g_pano = load_json(pano_dict_path)
        unavailable_dict = load_json(unavailable_path)
        g_unavail = set(unavailable_dict.keys())

    @staticmethod
    def _process_scene_streetview(scene_names, pano_views_out_path):
        """
        Process multiple scenes (small batches), returns {cam_token: {img_bytes, extrinsic, intrinsic, sample_token}}
        """
        part = {}
        name_to_scene = {s["name"]: s for s in g_nusc.scene}

        for scene_name in scene_names:
            scene = name_to_scene.get(scene_name)
            if scene is None:
                continue

            sample_token = scene["first_sample_token"]
            while sample_token != '':
                sample = g_nusc.get('sample', sample_token)
                lidar_token = sample['data']['LIDAR_TOP']
                sd_rec = g_nusc.get('sample_data', lidar_token)

                # Compatible with different nuScenes versions
                try:
                    ego_pose = g_nusc.get('ego_pose', sd_rec['token'])
                except KeyError:
                    ego_pose = g_nusc.get('ego_pose', sd_rec['ego_pose_token'])
                frame_id = ego_pose['token']

                if frame_id in g_unavail:
                    sample_token = sample['next']
                    continue

                # Load pano once, shared by all cameras in the same frame
                pano_id = g_frame[frame_id]['pano_id']
                pano_rel = g_pano[pano_id]['pano_path']
                pano_path = os.path.join(g_geo_root, pano_rel)
                pano = g_pano_img_cache.get(pano_path)
                if pano is None:
                    pano = cv2.imread(pano_path)  # BGR
                    if pano is None:
                        sample_token = sample['next']
                        continue
                    g_pano_img_cache[pano_path] = pano

                # Get frame and panorama lat/lon
                frame_lat = g_frame[frame_id]['lat']
                frame_lon = g_frame[frame_id]['lon']
                pano_lat = g_pano[pano_id]['lat']
                pano_lon = g_pano[pano_id]['lon']

                # Generate view for each camera
                cam_sensors = [k for k in sample['data'].keys() if 'CAM' in k]
                for cam in cam_sensors:
                    cam_token = sample['data'][cam]
                    cam_data = g_nusc.get('sample_data', cam_token)
                    cam_calib = g_nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                    cam_rot = Quaternion(cam_calib['rotation'])
                    
                    yaw, pitch = get_camera_yaw_pitch(g_nusc, cam_token)
                    view = extract_view_from_pano(pano=pano, yaw_deg=yaw, fov_h_deg=g_streetview_fov, out_size=g_streetview_size)

                    # Calculate extrinsic and intrinsic
                    x_global, y_global = get_xy_by_latlon(frame_lat, frame_lon, pano_lat, pano_lon)
                    z_global = 2.0

                    ego_rot = Quaternion(ego_pose['rotation'])
                    ego_rot_matrix = ego_rot.rotation_matrix
                    
                    translation_global = np.array([x_global, y_global, z_global])
                    translation_ego = ego_rot_matrix.T @ translation_global
                    
                    pano_extrinsic = np.eye(4)
                    pano_extrinsic[:3, :3] = cam_rot.rotation_matrix
                    pano_extrinsic[:3, 3] = translation_ego

                    pano_intrinsic = get_pano_intrinsic(g_streetview_size[0], g_streetview_size[1], g_streetview_fov)

                    ok, buf = cv2.imencode(".jpg", view)
                    if ok:
                        part[cam_token] = {
                            "img_bytes": bytes(buf),
                            "extrinsic": pano_extrinsic,
                            "intrinsic": pano_intrinsic,
                            "sample_token": sample['token']  # Add sample_token for folder structure
                        }

                sample_token = sample['next']

        return part

    @staticmethod
    def _mp_init_satellite(dataroot, version, geoext_root, sate_size, sat_orin_path):
        """Initialize satellite processing worker"""
        global g_nusc, g_geo_root, g_sate_size
        g_nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)
        g_geo_root = geoext_root
        g_sate_size = sate_size
        
        # Build gps_ranges_dict and load images in subprocess
        global g_gps_ranges, g_sat_imgs, g_sat_H
        g_gps_ranges = {}
        g_sat_imgs = {}
        g_sat_H = {}
        
        # Load satellite images in subprocess
        for key in gps_ranges.keys():
            image_filename = os.path.join(sat_orin_path, f"{key}.png")
            if os.path.exists(image_filename):
                try:
                    img = cv2.imread(image_filename)
                    if img is not None:
                        H, W = img.shape[:2]
                        # Copy gps_ranges info and add dimensions
                        g_gps_ranges[key] = gps_ranges[key].copy()
                        g_gps_ranges[key]["W"] = W
                        g_gps_ranges[key]["H"] = H
                        g_sat_imgs[key] = img
                        g_sat_H[key] = get_satellite_homography(sate_size, 0.15)
                except Exception as e:
                    print(f"Error loading {image_filename}: {e}")

    @staticmethod
    def _process_scene_satellite(scene_names, sat_out_path):
        """
        Process multiple scenes' satellite images, returns {lidar_sd_token: {img_bytes, H}}
        """
        part = {}
        name_to_scene = {s["name"]: s for s in g_nusc.scene}

        for scene_name in scene_names:
            scene = name_to_scene.get(scene_name)
            if scene is None:
                continue

            scene_token = scene['token']
            location = g_nusc.get('log', scene['log_token'])['location']
            poses = get_poses(g_nusc, scene_token)
            coordinates = derive_latlon(location, poses)

            for d in coordinates:
                loc = (d["longitude"], d["latitude"])
                rot = d["rot"]
                lidar_sd_token = d["token"]

                key, center_point = find_key_for_location(g_gps_ranges, loc)
                base_img = g_sat_imgs.get(key, None)
                if base_img is None:
                    continue

                corners = get_rectangle_corners(center_point, g_sate_size, rot)
                cropped = crop_quadrilateral(base_img, corners)  # BGR

                ok, buf = cv2.imencode(".png", cropped)
                if ok:
                    part[lidar_sd_token] = {
                        "img_bytes": bytes(buf),
                        "H": g_sat_H.get(key, None)
                    }

        return part

    def get(self, data_type: str, token: str):
        if "streetview" in data_type:
            if token in self.unavailable_dict:
                return None
            streetview_data = self.streetview_data.get(token, None)
            if streetview_data is None:
                return None
            if data_type == "streetview_data":
                img_or_path = streetview_data["streetview_img"]
                if isinstance(img_or_path, str):
                    img_path = os.path.join(self.geoext_dataroot, img_or_path)
                    return Image.open(img_path).convert("RGB")
                else:
                    # Already cached in memory
                    return img_or_path
            elif data_type == "streetview_path":
                return streetview_data["streetview_path"]
            elif data_type == "streetview_extrinsic":
                return streetview_data["streetview_extrinsic"]
            elif data_type == "streetview_intrinsic":
                return streetview_data["streetview_intrinsic"]
            else:
                raise ValueError(f"Unknown data type: {data_type}")

        elif "satellite" in data_type:
            # if token in self.unavailable_dict:
            #     return None
            satellite_data = self.satellite_data.get(token, None)
            if satellite_data is None:
                return None
            if data_type == "satellite_data":
                img_or_path = satellite_data["satellite_data"]
                if isinstance(img_or_path, str):
                    img_path = os.path.join(self.geoext_dataroot, img_or_path)
                    return Image.open(img_path).convert("RGB")
                else:
                    # Already cached in memory
                    return img_or_path
            elif data_type == "satellite_path":
                return satellite_data["satellite_path"]
            elif data_type == "satellite_pix2ego":
                return satellite_data["satellite_pix2ego"]
            else:
                raise ValueError(f"Unknown data type: {data_type}")
        else:
            raise ValueError(f"Unknown data type: {data_type}")



