"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.objaverse_ret_datasets import (
    ObjaverseRetDataset,
    ObjaverseRetEvalDataset,
)

from lavis.common.registry import registry


@registry.register_builder("objaverse_retrieval")
class ObjaverseRetBuilder(BaseDatasetBuilder):
    train_dataset_cls = ObjaverseRetDataset
    eval_dataset_cls = ObjaverseRetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/objaverse/defaults_ret.yaml"}
