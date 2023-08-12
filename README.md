# BLIP-3D

## Preparation

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
cd blip-3d
mkdir data
# objaverse
ln -s /path/to/objaverse
```

download **link** to`blip-3d/data/merged_data_new.json`

#### 3. convert dataset into training format

the abs path of converted dataset should be registered in `lavis/configs/default.yaml` as `cache_root`



## Training

```shell
$ conda activate lavis
# use facebook/opt-2.7b:
# stage 1:
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/pretrain_state1_point_obja.yaml
# stage 2:
$ python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_point_obja.yaml
```



## Evaluation

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
	...
]
```

