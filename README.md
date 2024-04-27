# <span style="color:lightblue">[CVPR2024]</span> GPT4Point<a> <img src="./readme_figs/icon.png" width="30" /> </a>: A Unified Framework for Point-Language Understanding and Generation


<p align="center">
  <a href="http://arxiv.org/abs/2312.02980" target='_**blank**'>
    <img src="https://img.shields.io/badge/arXiv paper-2312.02980📖-blue?">
  </a> 
  <a href="https://gpt4point.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
  <a href="https://gpt4point.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/version-v1.0-green">
  </a>
</p>

## 🔥 News
🔥 2024/04/27: We have modified the point encoder section, and now evaluation is more functional, although the training section still needs modification.

🔥 2024/04/13: We release the **GPT4Point** <span style="color:red">**v1.0**</span>, including training and 3D captioning evluation code.

🔥 2024/04/05:  Our paper **GPT4Point** is selected as **CVPR'24 Highlight** 2.84% (324/11532) !

🔥 2024/02/27:  Our paper **GPT4Point** is accepted by **CVPR'24**!

🔥 2024/01/19:  We release the **Objaverse-XL (Point Cloud Format)** Download and Extraction way.

🔥 2023/12/05:  The paper [GPT4Point (arxiv)](https://arxiv.org/abs/2312.02980) has been released, we unified the Point-language Understanding and Generation.

🔥 2023/08/13:  Two-stage Pre-training code of PointBLIP has been released.

🔥 2023/08/13:  Part of datasets used and result files has been uploaded.

## 🏠 Overview
<p align="center">  <a>  <img src="./readme_figs/fig1_teaser.png"  width="1000" /> </a> </p>

This project presents **GPT4Point**<a>  <img src="./readme_figs/icon.png"  width="20" /> </a>, a 3D multi-modality model that aligns **3D point clouds** with **language**. More details are shown in [project page](https://gpt4point.github.io/).

- **Unified Framework for Point-language Understanding and Generation.** We present the unified framework for point-language understanding and generation GPT4Point, including the 3D MLLM for point-text tasks and controlled 3D generation.

- **Automated Point-language Dataset Annotation Engine Pyramid-XL.** We introduce the automated point-language dataset annotation engine Pyramid-XL based on Objaverse-XL, currently encompassing 1M pairs of varying levels of coarseness and can be extended cost-effectively.

- **Object-level Point Cloud Benchmark.** Establishing a novel object-level point cloud benchmark with comprehensive evaluation metrics for 3D point cloud language tasks. This benchmark thoroughly assesses models' understanding capabilities and facilitates the evaluation of generated 3D objects.

## 🧭 Version
- **v1.0 (2024/04/13).** We release the training and evaluation (3D captioning) code.  
Dataset and text annotation: **Cap3D**.  
LLM Model: **OPT 2.7b**


## 🔧 Installation

1. (Optional) Creating conda environment

```bash
conda create -n gpt4point python=3.8
conda activate gpt4point
```

2. install from [PyPI](https://pypi.org/project/salesforce-lavis/)
```bash
pip install salesforce-lavis
```

3. Or, for development, you may build from source

```bash
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```
## 📦 Data Preparation
1. **Annotations**:
All annotations will be downloaded automaticly through hugging_face.

2. **Point Cloud**:
You can download the **Cap3D** point cloud dataset through the [Google Drive Link](https://drive.google.com/drive/folders/18uqvjVeEqVIWsZFHxoIXjb1LkZ9ZNTh0?usp=sharing). You should unzip these 10 tar.gz files and then put them together.
and the all folder strucure is:

```bash
GPT4Point
├── data
│   ├── cap3d
│   │   ├── points
│   │   │    ├── Cap3D_pcs_8192_xyz_w_color
│   │   │    │    ├── <point cloud id>.pkl
│   │   │    │    ├── ...
│   │   │    │    ├── <point cloud id>.pkl
│   │   ├── annotations
│   │   │    ├── cap3d_caption_train.json
│   │   │    ├── cap3d_caption_val.json
│   │   │    ├── cap3d_real_and_chatgpt_caption_test.json
│   │   │    ├── cap3d_real_and_chatgpt_caption_test_gt.json (for evaluation)
```

## 🚆 Training
1. For stage 1 training:
```bash
python -m torch.distributed.run --master_port=32339 --nproc_per_node=4 train.py --cfg-path lavis/projects/gpt4point/train/pretrain_stage1_cap3d.yaml
```

2. For stage 2 training:
```bash
python -m torch.distributed.run --master_port=32339 --nproc_per_node=4 train.py --cfg-path lavis/projects/gpt4point/train/pretrain_stage2_cap3d_opt2.7b.yaml
```

## 🏁 Evaluation
```bash
python -m torch.distributed.run --master_port=32239 --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/gpt4point/eval/captioning3d_cap3d_opt2.7b_eval.yaml
```


## 📦 Point Dataset and Data Annotation Engine (Optional)
### Objaverse-XL Point Dataset Download Way

**Note that you should cd in the Objaverse-xl_Download directory.**

```bash
cd ./Objaverse-xl_Download
```

Then please see the folder [Objaverse-xl_Download](./Objaverse-xl_Download) for details.


### Objaverse-XL Point Cloud Data Generation

Please see the [Extract_Pointcloud](./Objaverse-xl_Download/shap-e/) for details.

## 📝 TODO List
Dataset and Data Engine
- [✔] Release the arxiv and the project page.
- [✔] Release the dataset (Objaverse-Xl) Download way.
- [✔] Release the dataset (Objaverse-Xl) rendering (points) way.
- [✔] Release pretrain training code and 3D captioning val code.
- [ ] Release dataset and data annotation engine (Pyramid-XL). 
- [ ] Release more evaluation code.
- [ ] Release more trainingn code.
- [ ] Release more models.


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{GPT4Point,
  title={GPT4Point: A Unified Framework for Point-Language Understanding and Generation},
  author={Zhangyang Qi and Ye Fang and Zeyi Sun and Xiaoyang Wu and Tong Wu and Jiaqi Wang and Dahua Lin and Hengshuang Zhao},
  booktitle={CVPR},
  year={2024},
}
```


## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.



## 📚 Related Work
Together, Let's make LLM for 3D great!
- [Point-Bind & Point-LLM](https://arxiv.org/abs/2309.00615): It aligns point clouds with Image-Bind to reason multi-modality input without 3D-instruction data training.
- [3D-LLM](https://arxiv.org/abs/2307.12981): employs 2D foundation models to encode multi-view images of 3D point clouds.
- [PointLLM](https://arxiv.org/abs/2308.16911): employs 3D point clouds with LLaVA.
