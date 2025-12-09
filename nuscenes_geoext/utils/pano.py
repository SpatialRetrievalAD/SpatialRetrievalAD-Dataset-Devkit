import numpy as np
import cv2


def map_image_to_pano(img_bgr, fov_h_deg, yaw_deg, pano_width=3600, fill_value=0):
    """
    Map a single camera image to an equirectangular panorama slice and return the slice plus mask.
    """
    H, W = img_bgr.shape[:2]
    fov_h_rad = np.deg2rad(fov_h_deg)
    f_x = W / (2 * np.tan(fov_h_rad / 2))
    f_y = f_x  # Assume square pixels
    c_x = W / 2
    c_y = H / 2

    pano_h = pano_width // 2  # 2:1 equirectangular
    pano_w = pano_width

    pano_part = np.full((pano_h, pano_w, 3), fill_value, dtype=np.uint8)
    mask = np.zeros((pano_h, pano_w), dtype=np.uint8)

    theta = np.linspace(-np.pi, np.pi, pano_w, endpoint=False)  # [-pi, pi)
    phi = np.linspace(-np.pi/2, np.pi/2, pano_h)               # [-pi/2, pi/2]
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Yaw is subtracted to align heading
    theta_rel = theta_grid - np.deg2rad(yaw_deg)

    # Sphere -> camera coordinates
    X = np.cos(phi_grid) * np.sin(theta_rel)
    Y = np.sin(phi_grid)
    Z = np.cos(phi_grid) * np.cos(theta_rel)

    # Drop back-facing points (Z <= 0)
    valid = Z > 0

    # Project to camera plane
    u = f_x * (X / (Z + 1e-8)) + c_x
    v = f_y * (Y / (Z + 1e-8)) + c_y

    # Exclude bottom-left and bottom-right watermark blocks
    watermark_mask = (
        ((u >= 0) & (u < 80) & (v >= H - 50) & (v < H)) |
        ((u >= W - 50) & (u < W) & (v >= H - 50) & (v < H))
    )
    valid = valid & ~watermark_mask

    u_map = u.astype(np.float32)
    v_map = v.astype(np.float32)

    # Remap only the forward hemisphere
    part = cv2.remap(
        img_bgr,
        u_map,
        v_map,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value
    )

    pano_part[valid] = part[valid]
    mask[valid] = (np.any(part[valid] != fill_value, axis=1)).astype(np.uint8)

    # Erode mask to shrink edges and avoid black fringe pixels
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    meta = {"pano_w": pano_w, "pano_h": pano_h}
    return pano_part, mask, meta




def extract_view_from_pano(pano, yaw_deg, fov_h_deg, out_size=(640, 640), pitch_deg=0):
    """
    Extract a perspective view from an equirectangular panorama.
    Matches Google Street View API heading/pitch/fov semantics.

    Args:
        pano: equirectangular panorama (H, W, 3)
        yaw_deg: heading in degrees, clockwise [0-360]
        fov_h_deg: horizontal field of view
        out_size: output image size (w, h)
        pitch_deg: pitch in degrees, [-90, 90]
    """
    pano_h, pano_w = pano.shape[:2]
    out_w, out_h = out_size

    # Reconstruct full panorama if only the middle band was stored
    if pano_h < (pano_w // 2):
        background = np.zeros((pano_w // 2, pano_w, 3), dtype=np.uint8)
        background[pano_w//4 - pano_h//2:pano_w//4 + pano_h//2, :] = pano
        pano = background
        pano_h = pano_w // 2

    # Camera intrinsics for the output view
    fov_h_rad = np.deg2rad(fov_h_deg)
    f_x = out_w / (2 * np.tan(fov_h_rad / 2))
    f_y = f_x
    c_x = out_w / 2
    c_y = out_h / 2

    # Pixel grid in the target view
    xx, yy = np.meshgrid(np.arange(out_w), np.arange(out_h))
    X = (xx - c_x) / f_x
    Y = (yy - c_y) / f_y
    Z = np.ones_like(X)

    # Normalize rays
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X /= norm
    Y /= norm
    Z /= norm

    # Build rotation matrix (yaw, pitch)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Rotate around Y (yaw / heading)
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Rotate around X (pitch)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    R = R_pitch @ R_yaw

    # Apply rotation
    dirs = np.stack([X, Y, Z], axis=-1)
    dirs_rot = dirs @ R.T
    Xr, Yr, Zr = dirs_rot[...,0], dirs_rot[...,1], dirs_rot[...,2]

    # Convert to spherical angles
    theta = np.arctan2(Xr, Zr)     # [-pi, pi]
    phi = np.arcsin(np.clip(Yr, -1, 1))   # [-pi/2, pi/2]

    # Convert to equirectangular pixel coordinates
    u = (theta + np.pi) / (2 * np.pi) * pano_w
    v = (np.pi/2 - phi) / np.pi * pano_h

    u = u.astype(np.float32)
    v = v.astype(np.float32)

    view = cv2.remap(
        pano,
        u,
        v,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_WRAP  # Horizontal wrap
    )
    view = cv2.flip(view, 0) 

    return view
