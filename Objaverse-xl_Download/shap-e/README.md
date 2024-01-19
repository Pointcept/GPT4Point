# Extract Colorful PointClouds from Multi-view Images

![demo images](./pc_demo.png)

## Usage

1. Install with `pip install -e .`. Please make sure you download the shap-E code from this repo as there are modifications compared to the original repo. (requires python >= 3.9)

2. Run `python extract_pointcloud.py` to get colorful pointclouds from multi and the results will be saved at `./extracted_pts`. 

```
cd ./Objaverse-xl_Download/shap-e

python extract_pointcloud.py --mother_dir <path1> --cache_dir <path2> --save_name <path3>
```

> path1: your own path to store render zip. files. 
> 
> path2: directory to cache npz format data.
> 
> path3: output directory to store ply files.

## Acknowledgement

This code is modified based on [shapE](https://github.com/openai/shap-e/tree/main) and [Cap3D](https://github.com/crockwell/Cap3D/tree/main)