<div align="center">

# Spatial Retrieval Augmented Autonomous Driving (CVPR 2026)

## SpatialRetrievalAD Dataset & Devkit

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2512.06865/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://spatialretrievalad.github.io/)
[![Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/SpatialRetrievalAD/nuScenes-Geography-Data)

**Xiaosong Jia**<sup>1*</sup>, **Chenhe Zhang**<sup>1*</sup>, **Yule Jiang**<sup>2*</sup>, **Songbur Wong**<sup>2*</sup><br>**Zhiyuan Zhang**<sup>2</sup>, **Chen Chen**<sup>3</sup>, **Shaofeng Zhang**<sup>4</sup>, **Xuanhe Zhou**<sup>2</sup>, **Xue Yang**<sup>2†</sup><br> **Junchi Yan**<sup>2†</sup>, **Yu-Gang Jiang**<sup>1</sup>

<sup>1</sup>Institute of Trustworthy Embodied AI, Fudan University  
<sup>2</sup>Shanghai Jiao Tong University  
<sup>3</sup>Key Laboratory of Target Cognition and Application Technology, Aerospace Information Research Institute, Chinese Academy of Sciences  
<sup>4</sup>University of Science and Technology of China  

<small><sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding authors</small>
> 📧 Primary Contact: [Xiaosong Jia](https://jiaxiaosong1002.github.io/) (jiaxiaosong@fudan.edu.cn)

</div>



https://github.com/user-attachments/assets/c105f9cc-2439-4e37-b7fe-da71f61773f9


<br>

<div align="center">
  <img src="assets/teaser.jpg" width="800"/>
  <br>
</div>

<br>





## 🌍  Introduction <a name="introduction"></a>

This repository provides the official devkit for the nuScenes-Geography dataset introduced in our paper, **"Spatial Retrieval Augmented Autonomous Driving"**.

We introduce a novel **Spatial Retrieval Paradigm** that retrieves offline geographic images (Satellite/Streetview) based on GPS coordinates to enhance autonomous driving tasks. For multi-task learning, we design a plug-and-play **Spatial Retrieval Adapter** and a **Reliability Estimation Gate** to robustly fuse this external knowledge into model representations, followed retrieval injection mode of [Bench2Drive-R](https://arxiv.org/abs/2412.09647).


The following figure shows the spatial distribution and coverage status of our released [nuScenes-Geography dataset](https://huggingface.co/datasets/SpatialRetrievalAD/nuScenes-Geography-Data) across the nuScenes scenes. Please refer to our paper for a detailed description and analysis.

<div align="center">
  <img src="assets/distribution.jpg" width="800"/>
  <br>
</div>





## 📖 Table of Contents

- [Introduction](#introduction)
- [News](#news)
- [Multi-Task Implementations](#tasks)
- [Dataset & Devkit Installation](#install)
  - [Installing the Devkit](#install-devkit)
  - [Downloading the Dataset](#download-dataset)
- [Dataset Reconstruction](#construct)
- [Usage in Your Own Project](#usage)
- [Acknowledgments](#acknowledgments)


## 🔥 News <a name="news"></a>

- **[2025-12-09]** The **nuScenes-Geography** dataset and curation tools are released.





## 🚀 Multi-Task Implementations <a name="tasks"></a>

All implementation repositories are hosted under the **[SpatialRetrievalAD](https://github.com/SpatialRetrievalAD)** organization.




| Tasks                   |   Repositories   | 
| :----------------       | :----------------------:|
| Generative World Model  |  [Generative-World-Model](https://github.com/SpatialRetrievalAD/Generative-World-Model) |
| End-to-End Planning     |  [End2End-Planning](https://github.com/SpatialRetrievalAD/End2End-Planning) |
| Online Mapping | [Online Mapping](https://github.com/SpatialRetrievalAD/Online-Mapping) |
| Occupancy Prediction | [Occupancy-Prediction](https://github.com/SpatialRetrievalAD/Occupancy-Prediction)|
| 3D Detection | [3D-Detection](https://github.com/SpatialRetrievalAD/3D-Detection) |




## 📦 Dataset & Devkit Installation <a name="install"></a>


### 🛠️ Step 1: Installing the Devkit <a name="install-devkit"></a>

Clone the official devkit repository from GitHub and install it in editable mode:

```bash
git clone https://github.com/SpatialRetrievalAD/SpatialRetrievalAD-Dataset-Devkit.git
cd SpatialRetrievalAD-Dataset-Devkit
pip install -e .
```


### ⬇️ Step 2: Downloading the Dataset <a name="download-dataset"></a>

Download the dataset from Hugging Face:

👉 [SpatialRetrievalAD/nuScenes-Geography-Data](https://huggingface.co/datasets/SpatialRetrievalAD/nuScenes-Geography-Data)

```bash
hf download SpatialRetrievalAD/nuScenes-Geography-Data --repo-type=dataset
```

The dataset directory is organized as follows:

```
nuScenes-Geography-Data
├── frame_metadata.json
├── pano_metadata.json
├── unavailable_metadata.json
├── sat
│   ├── boston-seaport.png
│   ├── singapore-hollandvillage.png
│   ├── singapore-onenorth.png
│   └── singapore-queenstown.png
└── streetview
    ├── quality_labels.json
    └── panos
        ├── <pano_id_0>.jpg
        └── <pano_id_1>.jpg
```

The following figure show the correspondence between Geography images and nuScenes images:

<div align="center">
  <img src="assets/data_example.jpg" width="800"/>
  <br>
</div>



## 🔍 Usage in Your Own Project <a name="usage"></a>

Get started with the dataset by following the [Usage in Your Own Project](docs/usage.md) guide.


## 🔧 Dataset Reconstruction <a name="construct"></a>

For more details, please refer to [Dataset Reconstruction](docs/reconstruction.md)

<div align="center">
  <img src="assets/construction.jpg" width="800"/>
  <br>
</div>


## 🖊️ Citation
```
@inproceedings{jia2026spatial,
      title={Spatial Retrieval Augmented Autonomous Driving}, 
      author={Xiaosong Jia and Chenhe Zhang and Yule Jiang and Songbur Wong and Zhiyuan Zhang and Chen Chen and Shaofeng Zhang and Xuanhe Zhou and Xue Yang and Junchi Yan and Yu-Gang Jiang},
       booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
       year={2026}
}
```

## 🙏 Acknowledgments <a name="acknowledgments"></a>

We thank the following projects for their contributions to the development of this project: [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [FB-OCC](https://github.com/NVlabs/FB-BEV), [FlashOCC](https://github.com/Yzichen/FlashOCC), [MagicDriveDiT](https://github.com/flymin/MagicDrive-V2), [MapTR](https://github.com/hustvl/MapTR), [MapTRv2](https://github.com/hustvl/MapTR/tree/maptrv2), [nuScenes](https://www.nuscenes.org/), [PETR](https://github.com/megvii-research/PETR), [UniMLVG](https://github.com/SenseTime-FVG/OpenDWM), [VAD](https://github.com/hustvl/VAD)
