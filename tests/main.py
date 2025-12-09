from nuscenes.nuscenes import NuScenes
from nuscenes_geoext.nugeo import NuScenesGeoExt


if __name__ == '__main__':
    nusc = NuScenes(dataroot='path-to-nuScenes', version='v1.0-trainval', verbose=True)
    nugeo = NuScenesGeoExt(dataroot='path-to-nuScenes',
                           version='v1.0-trainval',
                           geoext_dataroot='path-to-nuScenes-Geography-Data',
                           streetview_size=(512, 512),
                           sate_size=(400, 400),
                           cache_flag=False) # Set to False to save memory

    # nusc = NuScenes(dataroot='path-to-nuScenes', version='v1.0-mini', verbose=True)
    # nugeo = NuScenesGeoExt(dataroot='path-to-nuScenes',
    #                        version='v1.0-mini',
    #                        geoext_dataroot='path-to-nuScenes-Geography-Data',
    #                        streetview_size=(512, 512),
    #                        sate_size=(400, 400),
    #                        cache_flag=False)  


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
                streetview_path = nugeo.get("streetview_path", cam_token)
                streetview_data = nugeo.get("streetview_data", cam_token)
                streetview_intrinsic = nugeo.get("streetview_intrinsic", cam_token)
                streetview_extrinsic = nugeo.get("streetview_extrinsic", cam_token)

            sample_token = sample['next']

