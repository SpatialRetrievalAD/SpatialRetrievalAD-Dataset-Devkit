# Dataset Reconstruction (Optional)

We recommend using our released dataset:

👉 **[SpatialRetrievalAD/nuScenes-Geography-Data](https://huggingface.co/datasets/SpatialRetrievalAD/nuScenes-Geography-Data)**

---

If you wish to **re-download Google Street View data** and fully **reconstruct the dataset**, you may use the script:

> `nuscenes_geoext/download_pano.py`

This process requires a valid **[Google API key](https://developers.google.com/maps/documentation/streetview/overview)** and access to the official **nuScenes** dataset.

---

## Usage Tutorial

### Step 1: Prepare Environment

Install the devkit and dependencies:

```bash
git clone https://github.com/SpatialRetrievalAD/SpatialRetrievalAD-Dataset-Devkit.git
cd SpatialRetrievalAD-Dataset-Devkit
pip install -e .
```



Ensure:

- nuScenes dataset is downloaded

- You have a valid [Google API key](https://developers.google.com/maps/documentation/streetview/overview)

write your API key to the `nuscenes_geoext/download_pano.py` file:

```python
GOOGLE_API_KEY = "your_api_key"
```

or pass the key directly in the command.

### Step 2: Run the Reconstruction Script

```bash
python nuscenes_geoext/download_pano.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --api_key YOUR_GOOGLE_API_KEY \
    --output_dir /path/to/output
```