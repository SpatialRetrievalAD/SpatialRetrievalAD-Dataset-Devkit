from pyquaternion import Quaternion
import numpy as np
import math
import cv2

# Identify which GPS range contains the location and return its key
def find_key_for_location(gps_ranges, location):
    lon, lat = location
    for key, range_info in gps_ranges.items():
        if (range_info["min"][0] <= lon and lon <= range_info["max"][0] and
                range_info["min"][1] <= lat and lat <= range_info["max"][1]):
            x = (lon - range_info["min"][0]) / (range_info["max"][0] - range_info["min"][0]) * range_info["W"]
            y = (1 - (lat - range_info["min"][1]) / (range_info["max"][1] - range_info["min"][1])) * range_info["H"]
            center_point  = (x,y)

            return key, center_point

    print("ERROR", location)
    return None

def get_rectangle_corners(center, size, rotation):
    """Calculate the corners of a rectangle given its center, size, and rotation."""
    W, H = size
    rotation_matrix = Quaternion(rotation).rotation_matrix[:2, :2].T
    half_x = np.array([W / 2, 0])
    half_y = np.array([0, H / 2])
    center = np.array([center[0], center[1]])
    rotated_half_x = np.dot(rotation_matrix, half_x)
    rotated_half_y = np.dot(rotation_matrix, half_y)
    
    corners = np.array([
        center - rotated_half_x - rotated_half_y,
        center + rotated_half_x - rotated_half_y,
        center + rotated_half_x + rotated_half_y,
        center - rotated_half_x + rotated_half_y
    ])
    return corners

def get_vehicle_coords_map(world_size, image_size=(400, 400)):
    """
    Calculate the coordinates of each pixel in the satellite image in the nuScenes vehicle coordinate system.
    
    nuScenes vehicle coordinate system definition:
    - Origin: vehicle center
    - x-axis: pointing forward (vehicle front)
    - y-axis: pointing left
    
    Args:
        world_size: tuple (W, H) - real-world size of the satellite image in meters
        image_size: tuple (width, height) - size of the cropped image in pixels, default (400, 400)
    
    Returns:
        coords_map: np.ndarray of shape (height, width, 2) - (x, y) coordinates in vehicle frame for each pixel (meters)
    """
    W_meter, H_meter = world_size
    width, height = image_size
    
    meter_per_pixel_x = W_meter / width
    meter_per_pixel_y = H_meter / height
    
    center_x = width / 2
    center_y = height / 2
    
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    dx_pixel = xx - center_x
    dy_pixel = yy - center_y
    
    # Map image axes to vehicle axes
    vehicle_x = -dy_pixel * meter_per_pixel_y
    vehicle_y = -dx_pixel * meter_per_pixel_x
    
    coords_map = np.stack([vehicle_x, vehicle_y], axis=-1)
    
    return coords_map

def crop_quadrilateral(img, corners):

    width = 400
    height = 400
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    corners = corners.astype(dst.dtype)
    # Perspective transform to crop
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped

def get_satellite_homography(image_size, pixel_size=0.15) -> np.ndarray:
    """
    Generate homography matrix to map satellite image pixels to vehicle coordinate frame.
    :param image_size: Tuple of (width, height) of the satellite image.
    :param pixel_size: Real-world size of each pixel in meters, default 0.15m.
    :return: Homography matrix (3x3 numpy array) mapping pixel coordinates to vehicle coordinates.
    """
    sat_width, sat_height = image_size
    cx = sat_width / 2.0
    cy = sat_height / 2.0
    s = pixel_size  # meter per pixel

    # Homography matrix from pixel (u,v,1) to vehicle (x,y,1)
    H = np.array([[s,   0.0, -s * cx],
                  [0.0, -s,  s * cy], 
                  [0.0, 0.0, 1.0]], dtype=np.float32),
    return H

