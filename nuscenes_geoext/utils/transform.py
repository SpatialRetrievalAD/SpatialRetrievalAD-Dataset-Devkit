import math
from typing import List, Dict, Tuple
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}

gps_ranges = {
    "boston-seaport": {
        "min": (-71.05851008588972, 42.33797303172644),
        "max": (-71.02355110151471, 42.35621253138901)
    },
    "singapore-hollandvillage": {
        "min": (103.7830192508019, 1.3045290611255334),
        "max": (103.80699190705191, 1.328495917110021)
    },
    "singapore-onenorth": {
        "min": (103.78398168303634, 1.2863845588122684),
        "max": (103.79696801116135, 1.3103515735653453)
    },
    "singapore-queenstown": {
        "min": (103.7687112761037, 1.2850861265683637),
        "max": (103.79268393235371, 1.3094289254199807)
    }
}


def get_poses(nusc: NuScenes, scene_token: str) -> List[dict]:
    """
    Return all ego poses for the current scene.
    :param nusc: The NuScenes instance to load the ego poses from.
    :param scene_token: The token of the scene.
    :return: A list of the ego pose dicts.
    """
    pose_list = []
    scene_rec = nusc.get('scene', scene_token)
    sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    
    ego_pose = nusc.get('ego_pose', sd_rec['token'])
    pose_list.append(ego_pose)

    while sd_rec['next'] != '':
        sd_rec = nusc.get('sample_data', sd_rec['next'])
        filename = sd_rec['filename']
        if filename.split('/')[0] == 'sweeps':
            continue
        ego_pose = nusc.get('ego_pose', sd_rec['token'])
        pose_list.append(ego_pose)

    return pose_list

def get_coordinate(ref_lat: float, ref_lon: float, bearing: float, dist: float) -> Tuple[float, float]:
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.
    :param ref_lat: Latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: Longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: The clockwise angle in radians between target point, reference point and the axis pointing north.
    :param dist: The distance in meters from the reference point to the target point.
    :return: A tuple of lat and lon.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS
    
    target_lat = math.asin(
        math.sin(lat) * math.cos(angular_distance) + 
        math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat)
    )
    return math.degrees(target_lat), math.degrees(target_lon)

def derive_latlon(location: str, poses: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    For each pose value, extract its respective lat/lon coordinate and timestamp.
    
    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).

    :param location: The name of the map the poses correspond to, ie: 'boston-seaport'.
    :param poses: All nuScenes egopose dictionaries of a scene.
    :return: A list of dicts (lat/lon coordinates and timestamps) for each pose.
    """
    assert location in REFERENCE_COORDINATES.keys(), \
        f'Error: The given location: {location}, has no available reference.'
    
    coordinates = []
    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    for p in poses:
        ts = p['timestamp']
        x, y = p['translation'][:2]
        bearing = math.atan(x / y) if y != 0 else (math.pi / 2 if x > 0 else -math.pi / 2)
        distance = math.sqrt(x**2 + y**2)
        lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)

        coordinates.append({
            'timestamp': ts, 
            'latitude': lat, 
            'longitude': lon, 
            'rot': p['rotation'], 
            'token': p['token']
        })
    return coordinates

def get_camera_yaw_pitch(nusc, sample_data_token):
    cam_data = nusc.get('sample_data', sample_data_token)
    calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

    # Intrinsics
    intrinsic = np.array(calib['camera_intrinsic'])
    fx = intrinsic[0, 0]

    # Extrinsics
    cam_rot = Quaternion(calib['rotation'])
    ego_rot = Quaternion(ego_pose['rotation'])
    cam_forward = cam_rot.rotate([0, 0, 1])  # Camera forward (z axis)

    cam_forward[0], cam_forward[1] = cam_forward[1], -cam_forward[0]  # Swap axes to match world frame
    world_forward = ego_rot.rotate(cam_forward)

    yaw = -np.arctan2(world_forward[1], world_forward[0]) * 180 / np.pi

    xy_norm = np.linalg.norm(world_forward[:2])
    pitch = np.arctan2(world_forward[2], xy_norm) * 180 / np.pi

    return yaw, pitch


def get_camera_relative_yaw(nusc, sample_data_token):
    """
    Compute camera yaw relative to the vehicle frame (degrees, 0-360).
    """
    cam_data = nusc.get('sample_data', sample_data_token)
    calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    cam_rot = Quaternion(calib['rotation'])
    cam_forward = [0, 0, 1]
    vehicle_forward = cam_rot.rotate(cam_forward)
    
    yaw = np.arctan2(vehicle_forward[1], vehicle_forward[0]) * 180 / np.pi
    yaw = -yaw
    return yaw


def get_pano_intrinsic(width: int, height: int, fov_h_deg: float) -> np.ndarray:
    """
    Calculate the intrinsic matrix for a panoramic camera.
    :param width: The width of the image.
    :param height: The height of the image.
    :param fov_h_deg: The horizontal field of view in degrees.
    :return: The intrinsic matrix.
    """
    fov_h_rad = np.deg2rad(fov_h_deg)
    f_x = width / (2 * np.tan(fov_h_rad / 2))
    f_y = f_x
    c_x = width / 2
    c_y = height / 2
    return np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

def get_xy_by_latlon(base_lat: float, base_lon: float, target_lat: float, target_lon: float) -> Tuple[float, float]:
    """
    Calculate the translation distances x and y from the base point to the target point.
    :param base_lat: Latitude of the base point in degrees.
    :param base_lon: Longitude of the base point in degrees.
    :param target_lat: Latitude of the target point in degrees.
    :param target_lon: Longitude of the target point in degrees.
    :return: A tuple containing the translation distances x and y in meters.
    """
    base_lat, base_lon, target_lat, target_lon = map(math.radians, [base_lat, base_lon, target_lat, target_lon])
    dlat = target_lat - base_lat
    dlon = target_lon - base_lon
    y = dlat * EARTH_RADIUS_METERS
    x = dlon * EARTH_RADIUS_METERS * math.cos(base_lat)
    return x, y
