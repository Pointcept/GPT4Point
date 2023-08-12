"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random
import numpy as np
import torch

from lavis.datasets.transforms.transforms_point import pc_norm, random_point_dropout, random_scale_point_cloud, shift_point_cloud, rotate_perturbation_point_cloud, rotate_point_cloud

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, pts_processor, vis_root, pts_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        pts_root (string): Root directory of images (e.g. coco/points/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, pts_processor, vis_root, pts_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            if "image_id" in ann:
                img_id = ann["image_id"]
            if "model_id" in ann and "taxonomy_id" in ann:
                img_id = ann["taxonomy_id"] + '-' + ann["model_id"]

            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        # image:
        # image_path = os.path.join(self.vis_root, ann["image"])
        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # text
        names = ann["name"]
        names = [name.strip() for name in names.split(',') if name.strip()]   # ['bag', 'traveling bag', 'travelling bag', 'grip', 'suitcase']
        name = random.choice(names) 
        text = 'an 3D point cloud of ' + name
        caption = self.text_processor(text)

        # point
        point = np.load(os.path.join(ann['point'])).astype(np.float32)                      # (8192, 3)
        point = pc_norm(point)
        point = random_point_dropout(point[None, ...])                                    # (1, 8192, 3) -> (1, 8192, 3)
        point = random_scale_point_cloud(point)
        point = shift_point_cloud(point)
        point = rotate_perturbation_point_cloud(point)
        point = rotate_point_cloud(point)
        point = point.squeeze()                                                           # (1, 8192, 3) -> (8192, 3)
        point = torch.from_numpy(point).float()

        return {
            "image": image,
            "text_input": caption,
            "point": point,
            # "image_id": self.img_ids[ann["image_id"]],
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, pts_processor, vis_root, pts_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, pts_processor, vis_root, pts_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
