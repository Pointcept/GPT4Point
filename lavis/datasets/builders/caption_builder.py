"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
# from lavis.datasets.datasets.coco_caption_datasets import (
#     COCOCapDataset,
#     COCOCapEvalDataset,
#     NoCapsEvalDataset,
# )

# from lavis.common.registry import registry
# from lavis.datasets.datasets.video_caption_datasets import (
#     VideoCaptionDataset,
#     VideoCaptionEvalDataset,
# )

# from lavis.datasets.datasets.shapenet_caption_datasets import (
#     ShapeNetCapDataset,
#     ShapeNetCapEvalDataset,
#     NoCapsEvalDataset,
# )

from lavis.datasets.datasets.objaverse_caption_datasets import (
    ObjaverseCapDataset,
    ObjaverseCapDataset_tune,
    ObjaverseCapEvalDataset,
)
from lavis.common.registry import registry
# @registry.register_builder("coco_caption")
# class COCOCapBuilder(BaseDatasetBuilder):
#     train_dataset_cls = COCOCapDataset
#     eval_dataset_cls = COCOCapEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/coco/defaults_cap.yaml",
#     }


# @registry.register_builder("nocaps")
# class COCOCapBuilder(BaseDatasetBuilder):
#     eval_dataset_cls = NoCapsEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/nocaps/defaults.yaml",
#     }


# @registry.register_builder("msrvtt_caption")
# class MSRVTTCapBuilder(BaseDatasetBuilder):
#     train_dataset_cls = VideoCaptionDataset
#     eval_dataset_cls = VideoCaptionEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/msrvtt/defaults_cap.yaml",
#     }


# @registry.register_builder("msvd_caption")
# class MSVDCapBuilder(BaseDatasetBuilder):
#     train_dataset_cls = VideoCaptionDataset
#     eval_dataset_cls = VideoCaptionEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/msvd/defaults_cap.yaml",
#     }


# @registry.register_builder("vatex_caption")
# class VATEXCapBuilder(BaseDatasetBuilder):
#     train_dataset_cls = VideoCaptionDataset
#     eval_dataset_cls = VideoCaptionEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/vatex/defaults_cap.yaml",
#     }

# @registry.register_builder("shapenet_caption")
# class ShapeNetCapBuilder(BaseDatasetBuilder):
#     train_dataset_cls = ShapeNetCapDataset
#     eval_dataset_cls = ShapeNetCapEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/shapenet/defaults_cap.yaml",
#     }

@registry.register_builder("objaverse_caption")
class ObjaverseCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ObjaverseCapDataset
    eval_dataset_cls = ObjaverseCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/defaults_cap.yaml",
    }

@registry.register_builder("objaverse_caption_tune")
class ObjaverseCap_tune_Builder(BaseDatasetBuilder):
    train_dataset_cls = ObjaverseCapDataset_tune
    eval_dataset_cls = ObjaverseCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/cap_tune.yaml",
    }