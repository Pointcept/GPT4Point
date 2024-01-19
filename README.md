# GPT4Point<a>  <img src="./readme_figs/icon.png"  width="30" /> </a>: A Unified Framework for Point-Language Understanding and Generation

<p align="center">
  	<a href="https://img.shields.io/badge/version-v0.1.0-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
  <a >
       <img alt="Status-building" src="https://img.shields.io/badge/Status-building-blue" />
  	</a>
  <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  	</a>
    <br />
</p>

## 🔥 News

🔥 2024/01/10:  We release the **Objaverse-XL (Point Cloud Format)** Download way.

🔥 2023/12/05:  The paper [GPT4Point (arxiv)](https://arxiv.org/abs/2312.02980) has been released, we unified the Point-language Understanding and Generation.

🔥 2023/08/13:  Two-stage Pre-training code of PointBLIP has been released.

🔥 2023/08/13:  Part of datasets used and result files has been uploaded.

## 🏠 Overview
<p align="center">  <a>  <img src="./readme_figs/fig1_teaser.png"  width="1000" /> </a> </p>

This project presents **GPT4Point**<a>  <img src="./readme_figs/icon.png"  width="20" /> </a>, a 3D multi-modality model that aligns **3D point clouds** with **language**. More details are shown in [project page](https://gpt4point.github.io/).

- **Unified Framework for Point-language Understanding and Generation.** We present the unified framework for point-language understanding and generation GPT4Point, including the 3D MLLM for point-text tasks and controlled 3D generation.

- **Automated Point-language Dataset Annotation Engine Pyramid-XL.** We introduce the automated point-language dataset annotation engine Pyramid-XL based on Objaverse-XL, currently encompassing 1M pairs of varying levels of coarseness and can be extended cost-effectively.

- **Object-level Point Cloud Benchmark.** Establishing a novel object-level point cloud benchmark with comprehensive evaluation metrics for 3D point cloud language tasks. This benchmark thoroughly assesses models' understanding capabilities and facilitates the evaluation of generated 3D objects.

## 📦 Point Dataset and Data Annotation Engine
### Objaverse-XL Point Dataset Download Way

**Note that you should cd in the Objaverse-xl_Download directory.**

```bash
cd ./Objaverse-xl_Download
```

Then please see the [Objaverse-xl_Download.md](./Objaverse-xl_Download/Objaverse-xl_Download.md) and the folder [Objaverse-xl_Download](./Objaverse-xl_Download) for details.


### Objaverse-XL Point Cloud Data Generation

Please see the [Extract_Pointcloud](./Objaverse-xl_Download/shap-e/) for details.

## 📝 TODO List
Dataset and Data Engine
- [✔] Release the arxiv and the project page.
- [✔] Release the dataset (Objaverse-Xl) Download way.
- [✔] Release the dataset (Objaverse-Xl) rendering (points) way. (2023.01.15)
- [ ] Release dataset and data annotation engine (Pyramid-XL).
- [ ] Add inferencing codes with checkpoints.
- [ ] Add Huggingface Demo🤗.
- [ ] Add training codes.
- [ ] Add evaluation codes.
- [ ] Add gradio demo codes.


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{qi2023gpt4point,
  title={GPT4Point: A Unified Framework for Point-Language Understanding and Generation}, 
  author={Zhangyang Qi and Ye Fang and Zeyi Sun and Xiaoyang Wu and Tong Wu and Jiaqi Wang and Dahua Lin and Hengshuang Zhao},
  year={2023},
  eprint={2312.02980},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
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
