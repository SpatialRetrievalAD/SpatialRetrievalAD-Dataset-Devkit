from setuptools import setup, find_packages


setup(
    name="nuscenes-geoext", 
    version="0.1.0",
    
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy",
        "opencv-python", 
        "pyquaternion",
        "nuscenes-devkit",
        "Pillow", 
        "requests",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "nuscenes_geoext": ["data/*.json"]
    }
)