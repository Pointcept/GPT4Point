"""
 Copyright (c) 2023, pjlab.
 All rights reserved.
"""

import os
# import sys
# sys.path.append('.')
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
from lavis.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate

from lavis.common.registry import registry
import numpy as np
from torchvision.datasets.utils import download_url
# from lavis.datasets.transforms.transforms_point import pc_norm, random_point_dropout, random_scale_point_cloud, shift_point_cloud, rotate_perturbation_point_cloud, rotate_point_cloud

from others.ptstext_benchmark.ptstext_eval import Ptstext_EvalCap
from others.ptstext_benchmark.ptstext_data_cocostyle import Ptstext_Benchmark
from huggingface_hub import hf_hub_download
import pickle
import io

class Cap3d_Captioning3d_Dataset(BaseDataset):
    def __init__(self, text_processor, pts_processor, pts_root, ann_paths, args=None):
        """
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(text_processor=text_processor, pts_processor=pts_processor, pts_root=pts_root, ann_paths=ann_paths)

        self.pcd_ids = {}
        n = 0
        for ann in self.annotation:
            pcd_id = ann["point"]
            if pcd_id not in self.pcd_ids.keys():
                self.pcd_ids[pcd_id] = n
                n += 1
        if hasattr(args, 'petrel_client'):      # it is for cluster in pjlab.
            self.petrel_client = args.petrel_client
            if self.petrel_client is True:
                from petrel_client.client import Client
                conf_path = '~/petreloss.conf'
                self.client = Client(conf_path)


    def __getitem__(self, index):
        ann = self.annotation[index]

        if 'pcd_id' in ann:
            point_id = ann['pcd_id']
        else:
            point_id = ann['point'].split('/')[-1][:-4]
        point_path = ann['point']
        if point_path.endswith('.npz'):
            if hasattr(self, 'petrel_client'):
                if self.petrel_client is True:
                    pcd_url = 's3://lavis/objaverse/objaverse_pc_parallel/' + point_id + '/' + point_id + '.npz'
                    binary_data = self.client.get(pcd_url)
                    npz_file = np.load(io.BytesIO(binary_data), allow_pickle=True)
                    point = np.array(pickle.load(open(npz_file, 'rb')))
            else:
                point = np.load(point_path)['arr_0'].astype(np.float32)
        elif point_path.endswith('.pkl'):
            if hasattr(self, 'petrel_client'):
                if self.petrel_client is True:
                    pcd_url = 's3://lavis/Cap3D_pcs_8192_xyz_w_color/' + point_id + '.pkl'
                    binary_data = self.client.get(pcd_url)
                    binary_data_io = io.BytesIO(binary_data)
                    point = np.array(pickle.load(binary_data_io))
            else:
                if 'point_cloud' in point_path:
                    point_path = point_path.replace('point_cloud', 'points')
                point = np.array(pickle.load(open(point_path, 'rb')))                         # (8192, 6)

        # point color
        if point.shape[-1] == 3:
            white_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # white rgb
            white_colors = np.tile(white_color, (point.shape[0], 1))
            point = np.concatenate((point, white_colors), axis=1)

        # point transform
        point = self.pts_processor(point)

        caption = self.text_processor(ann["caption"])
        return {
            "text_input": caption,
            "point": point
        }

class Cap3d_Captioning3d_EvalDataset(BaseDataset):
    def __init__(self, text_processor, pts_processor, pts_root, ann_paths, args=None):
        """
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(text_processor=text_processor, pts_processor=pts_processor, pts_root=pts_root, ann_paths=ann_paths)

        if hasattr(args, 'petrel_client'):
            self.petrel_client = args.petrel_client
            if self.petrel_client is True:
                from petrel_client.client import Client
                conf_path = '~/petreloss.conf'
                self.client = Client(conf_path)

    def __getitem__(self, index):
        ann = self.annotation[index]

        if 'pcd_id' in ann:
            point_id = ann['pcd_id']
        else:
            point_id = ann['point'].split('/')[-1][:-4]
        point_path = ann['point']
        if point_path.endswith('.npz'):
            if hasattr(self, 'petrel_client'):
                if self.petrel_client is True:
                    pcd_url = 's3://lavis/objaverse/objaverse_pc_parallel/' + point_id + '/' + point_id + '.npz'
                    binary_data = self.client.get(pcd_url)
                    npz_file = np.load(io.BytesIO(binary_data), allow_pickle=True)
                    point = np.array(pickle.load(open(npz_file, 'rb')))
            else:
                point = np.load(point_path)['arr_0'].astype(np.float32)
        elif point_path.endswith('.pkl'):   # it is .pkl
            if hasattr(self, 'petrel_client'):
                if self.petrel_client is True:
                    pcd_url = 's3://lavis/Cap3D_pcs_8192_xyz_w_color/' + point_id + '.pkl'
                    binary_data = self.client.get(pcd_url)
                    binary_data_io = io.BytesIO(binary_data)
                    point = np.array(pickle.load(binary_data_io))
            else:
                if 'point_cloud' in point_path:
                    point_path = point_path.replace('point_cloud', 'points')
                point = np.array(pickle.load(open(point_path, 'rb')))                         # (8192, 6)

        # point color, if it is for objaverse. it has no color.
        if point.shape[-1] == 3:
            white_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # white rgb
            white_colors = np.tile(white_color, (point.shape[0], 1))
            point = np.concatenate((point, white_colors), axis=1)
                
        # point and text transform
        point = self.pts_processor(point)
        caption = self.text_processor.prompt # 'a 3D point cloud of '

        pcd_id = ann['point'].split('/')[-1][:-4]
        return {
            "pcd_id": pcd_id,
            "point": point,
            "text_input": caption,
        }


def cap3d_captioning3d_eval(results_file, split):
    filepaths = {
        "test": "data/cap3d/annotations/cap3d_real_and_chatgpt_caption_test_gt.json",
    }
    if not os.path.exists(filepaths[split]):
        gt_directory, gt_filename = os.path.split(filepaths[split])
        hf_hub_download(repo_id="alexzyqi/GPT4Point", filename=filename, repo_type="dataset", local_dir=gt_directory)

    annotation_file = filepaths[split]

    ptstext_benchmark = Ptstext_Benchmark(annotation_file)
    ptstext_result = ptstext_benchmark.loadRes(results_file)
    ptstext_evalcap = Ptstext_EvalCap(ptstext_benchmark, ptstext_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    ptstext_evalcap.evaluate()

    # print output evaluation scores
    for metric, score in ptstext_evalcap.eval.items():
        print(f"{metric}: {score:.3f}")

    return ptstext_evalcap