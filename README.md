# PointBLIP

This project presents **PointBLIP**<a>  <img src="./figure/cloud.png"  width="15" /> </a>, a 3D multi-modality model that aligns **3D point clouds** with **language** inspired by 2D-multimodal model BLIP. 

- We directly align the representation of point cloud and language without the need to align with image modality additionally as classical methods. 
- Furthermore, we explore the great potential of point clouds for various tasks on 3D multimodal understanding and generation.




## Overview

<demo> 



## News

🔥 2023/08/13:  Two-stage Pretraining code of PointBind has been released.

🔥 2023/08/13:  All datasets used and result files has been uploaded.







## PointBLIP

### Previous method ULIP：
<p align="center"> <a>  <img src="./figure/ulip.png"  width="900" /> </a> </p>


### Our PointBLIP：

<p align="center">  <a>  <img src="./figure/pointblip.png"  width="780" /> </a> </p>

*ULIP* is a representative work on aligning point clouds with other modality information (Upper part). However, it needs to align 3D point clouds with both images and texts during training just to make the model gain the ability of 3D semantic understanding. 

To simplify this approach, our ${\ PointBLIP\ }$ considers directly aligning texts with 3D point clouds (Lower part). Besides, we add an LLM(Large Language Model) to the basis of joint representation learning, which fully promote the combination of 3D point cloud and text representation, and successfully apply to multiple downstream tasks.

Our PointBLIP exhibits 3 main characters:

- $\color{darkorange}{Directly\ Align\ Texts\ with\ 3D\ Point\ Clouds\ .}$ To improve the recognition ability and semantic understanding of 3D backbone models,   we directly align the representation of 3D point clouds and texts. We doesn't introduce additional infomation of image representions during training, which simplifies the training process and fully aligns the representations.
- $\color{darkorange}{Bridge\ Modality\ Gap\ Guided\ By\ BLIP2\ .}$ Inspired by 2D multi-modality model *BLIP2*, we ingeniously utilize both pretrained 3D point cloud models and large language models. We bridge the modality gap between 3D point clouds and texts using a trainable module (text encoder in the figure) pretrained in two-stages.
- $\color{darkorange}{LLM\ Empowers\ a\ Wide\ Range\ of\ 3D\ semantic\ tasks\ .}$  Incorporating the large language model enables the capability to perform a broader spectrum of 3D semantic understanding and genaration tasks. Besides engaging in a 3D classification task directly using the trained representations, ***PointBLIP*** can perform 3D caption generation, 3D retrieval and 3D question answering tasks, fully exploring the semantic capabilities of 3D point clouds





## 3D Caption





## Point Cloud QA





## Zero-Shot 3D classification







## Get Started

### Preparation

#### 1. Install [salesforce-lavis](https://github.com/salesforce/LAVIS)

```shell
$ conda create -n lavis python=3.8
$ conda activate lavis

$ git clone https://github.com/salesforce/LAVIS.git SalesForce-LAVIS
$ cd SalesForce-LAVIS
$ pip install -e .

$ pip install positional_encodings
```

#### 2. Prepare the dataset

```shell
git clone <Our repo>
cd blip-3d
mkdir data
# objaverse
ln -s /path/to/objaverse
```

download **link** to `blip-3d/data/merged_data_new.json`

#### 3. convert dataset into training format

the abs path of converted dataset should be registered in `lavis/configs/default.yaml` as `cache_root`



### Training

```shell
$ conda activate lavis
# use facebook/opt-2.7b:
# stage 1:
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/pretrain_state1_point_obja.yaml
# stage 2:
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_point_obja.yaml
```



### Evaluation

```shell
$ python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip2/eval/caption_objaverse_opt2.7b_eval.yaml
```

result will be saved as `.json` file in `lavis/output` with following formats:

```json
[
    {
        "image_id": "object hash id of objaverse",
        "2d_caption": "gt caption when training BLIP-3D",
        "caption": "generated caption by BLIP-3D"
    },
    
]
```
