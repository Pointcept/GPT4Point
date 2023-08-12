"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate

import numpy as np
from lavis.datasets.transforms.transforms_point import pc_norm, random_point_dropout, random_scale_point_cloud, shift_point_cloud, rotate_perturbation_point_cloud, rotate_point_cloud

class ObjaverseCapDataset(Dataset):
    def __init__(self, vis_processor, text_processor, pts_processor, vis_root, pts_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        self.vis_root = vis_root
        self.pts_root = "data/objaverse/objaverse_pc_parallel/" # use relative path

        with open('data/objaverse/common_ids.txt', 'r') as txt_file:
            self.annotation = [line.strip() for line in txt_file]
        
        text_json_path = 'data/merged_data_new.json' 
        with open(text_json_path, 'r') as json_file: # Read the content of the JSON file 
            self.text_json = json.load(json_file)

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.pts_processor = pts_processor

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def __getitem__(self, index):
        ann_id = self.annotation[index]
        point_path = os.path.join(self.pts_root, ann_id, f"{ann_id}_8192.npz")

        # point
        point = np.load(point_path)['arr_0'].astype(np.float32)                           # (8192, 3)
        point = pc_norm(point)                                                            # max: 274, min: -15.9 -> min: -0.49231547, max: 0.9409862

        # point augment
        point = random_point_dropout(point[None, ...])                                    # (1, 8192, 3) -> (1, 8192, 3)
        point = random_scale_point_cloud(point)
        point = shift_point_cloud(point)
        point = rotate_perturbation_point_cloud(point)
        point = rotate_point_cloud(point)
        point = point.squeeze()                                                           # (1, 8192, 3) -> (8192, 3)

        caption = self.text_json[ann_id]
        return {
            "text_input": caption,
            "point": point
        }

class ObjaverseCapDataset_tune(Dataset):
    def __init__(self, vis_processor, text_processor, pts_processor, vis_root, pts_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        self.vis_root = vis_root # '/nvme/datasets/lavis/objaverse/images/'
        self.pts_root = pts_root # '/nvme/datasets/lavis/objaverse/points/'

        self.ann2caption = json.load(open('data/overal_description_merged.json', 'r'))
        # filter color
        colors = ['orange', 'red', 'blue', 'green', 'purple', 'yellow', 'grey', 'white', 'pink']
        for k in self.ann2caption.keys():
            for color in colors:
                self.ann2caption[k] = self.ann2caption[k].replace(" " + color, "")
        self.annotation = list(self.ann2caption.keys())

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.pts_processor = pts_processor

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def __getitem__(self, index):
        ann_id = self.annotation[index]
        point_path = os.path.join(self.pts_root, ann_id, f"{ann_id}_8192.npz")

        # point
        point = np.load(point_path)['arr_0'].astype(np.float32)                           # (8192, 3)
        point = pc_norm(point)                                                            # max: 274, min: -15.9 -> min: -0.49231547, max: 0.9409862

        # point augment
        point = random_point_dropout(point[None, ...])                                    # (1, 8192, 3) -> (1, 8192, 3)
        point = random_scale_point_cloud(point)
        point = shift_point_cloud(point)
        point = rotate_perturbation_point_cloud(point)
        point = rotate_point_cloud(point)
        point = point.squeeze()                                                           # (1, 8192, 3) -> (8192, 3)

        caption = self.ann2caption[ann_id]
        return {
            "text_input": caption,
            "point": point
        }

class ObjaverseCapEvalDataset(Dataset):
    def __init__(self, vis_processor, text_processor, pts_processor, vis_root, pts_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        self.vis_root = vis_root # '/nvme/datasets/lavis/shapenet/images/'
        self.pts_root = pts_root # '/nvme/datasets/lavis/shapenet/points/'

        # with open('/nvme/datasets/objaverse/common_ids.txt', 'r') as txt_file:
        #     self.annotation = [line.strip() for line in txt_file]
        
        # text_json_path = '/home/qizhangyang/others/objaverse/merged_data_new.json' 
        # with open(text_json_path, 'r') as json_file: # Read the content of the JSON file 
        #     self.text_json = json.load(json_file)
        # fintune on 5w
        self.text_json = json.load(open('data/overal_description_merged.json', 'r'))
        # filter color
        colors = ['orange', 'red', 'blue', 'green', 'purple', 'yellow', 'grey', 'white', 'pink']
        for k in self.text_json.keys():
            for color in colors:
                self.text_json[k] = self.text_json[k].replace(" " + color, "")
        self.annotation = list(self.text_json.keys())

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.pts_processor = pts_processor

    def collater(self, samples):
        return default_collate(samples)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann_id = self.annotation[index]
        point_path = os.path.join(self.pts_root, ann_id, f"{ann_id}_8192.npz")

        # point
        point = np.load(point_path)['arr_0'].astype(np.float32)                           # (8192, 3)
        point = pc_norm(point)                                                            # max: 274, min: -15.9 -> min: -0.49231547, max: 0.9409862

        caption = self.text_processor(self.text_json[ann_id])
        return {
            "text_input": caption,
            "point": point,
            "ann_id": ann_id
        }



class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }
