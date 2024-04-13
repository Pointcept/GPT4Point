"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry

from lavis.datasets.datasets.cap3d_captioning3d_dataset import (
    Cap3d_Captioning3d_Dataset,
    Cap3d_Captioning3d_EvalDataset,
)

@registry.register_builder("cap3d_captioning3d")
class Cap3d_Captioning3d_Builder(BaseDatasetBuilder):
    train_dataset_cls = Cap3d_Captioning3d_Dataset
    eval_dataset_cls = Cap3d_Captioning3d_EvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cap3d/defaults_cap3d.yaml",
    }
