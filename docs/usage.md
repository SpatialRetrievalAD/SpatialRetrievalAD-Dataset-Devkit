

# Usage in Your Own Project


## 📑 Data Loading

We adopt a devkit interface consistent with nuScenes to facilitate dataset management:

```python
from nuscenes.nuscenes import NuScenes
from nuscenes_geoext.nugeo import NuScenesGeoExt

# v1.0-trainval
nusc = NuScenes(dataroot='path-to-nuscenes', version='v1.0-trainval', verbose=True)
nugeo = NuScenesGeoExt(dataroot='path-to-nuscenes',
                       version='v1.0-trainval',
                       geoext_dataroot='path-to-nuscenes-nuScenes-Geography-Data',
                       streetview_size=(512, 512),
                       sate_size=(400, 400),
                       cache_flag=False)
```




The first invocation of `NuScenesGeoExt` will generate all geography image tiles and create corresponding cache files. This initialization is therefore relatively time-consuming. Subsequent runs will directly reuse the cached tiles without regeneration.


The default value of `cache_flag` is False. This parameter controls whether cropped tile images are stored in the cached `.pkl` files. When disabled (default), the cache only records file paths to the tiles rather than embedding the image content.


Upon the first invocation of `NuScenesGeoExt`, the initialized dataset directory structure is organized as follows:

```
nuScenes-Geography-Data
├── frame_metadata.json
├── pano_metadata.json
├── unavailable_metadata.json
├── satellite_data_v1.0-trainval.pkl
├── streetview_data_v1.0-trainval.pkl
├── sat/
├── sat_slice/
└── streetview
    ├── quality_labels.json
    ├── cams/
    └── panos/
```

We recommend integrating this initialization step into the data preprocessing pipeline of your project—synchronized with nuScenes preprocessing—e.g., within scripts such as `create_data.py`. You may also refer to our open-source implementation for detailed examples.  [Multi-Task Implementations](#tasks)

## 🔎 Data Access

### Street View Image Retrieval

Each street view image corresponds to a camera frame of every nuScenes sample. For available street views, the API returns either a PIL image or its file path; if unavailable, it returns `None`.

```python 
sample = nusc.get('sample', sample_token)
cam_token = sample['data']['CAM_FRONT']

streetview_image = nugeo.get("streetview_data", cam_token)
streetview_path = nugeo.get("satellite_path", ego_pose_token)
```

Street view Intrinsics and Extrinsics

```python 
intrinsic = nugeo.get("streetview_intrinsic", cam_token)
extrinsic = nugeo.get("streetview_extrinsic", cam_token)
```

### Satellite Image Retrieval

Each satellite image corresponds to a nuScenes sample. For available satellite imagery, the API returns either a PIL image or its file path.

```python
sample = nusc.get('sample', sample_token)
ego_pose_token = nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token']

satellite_image = nugeo.get("satellite_data", ego_pose_token) 
satellite_path = nugeo.get("satellite_path", ego_pose_token)
```

Satellite homography

```python
satellite_pix2ego = nugeo.get("satellite_pix2ego", ego_pose_token)
```


### Example: Full Traversal Example

The following code snippet provides a complete traversal procedure for reference:

```python
camera_types = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"
]
for scene in nusc.scene:
    print(scene)
    scene_name = scene['name']
    scene_token = scene['token']
    sample_token = scene['first_sample_token']
    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        ego_pose_token = nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token']
        sat_data = nugeo.get("satellite_data", ego_pose_token)
        sat_path = nugeo.get("satellite_path", ego_pose_token)
        sat_pix2ego = nugeo.get("satellite_pix2ego", ego_pose_token)
        
        for cam in camera_types:
            cam_token = sample['data'][cam]
            street_path = nugeo.get("streetview_path", cam_token)
            street_data = nugeo.get("streetview_data", cam_token)
            street_intrinsic = nugeo.get("streetview_intrinsic", cam_token)
            street_extrinsic = nugeo.get("streetview_extrinsic", cam_token)

        sample_token = sample['next']
```

