"""
 Copyright (c) 2023, PJLAB.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.processors.base_processor import BaseProcessor
from lavis.processors.blip_processors import (
    BlipCaptionProcessor,
)
from lavis.processors.gpt4point_processors import (
    GPT4Point_Cap3D_Train_Processor,
    GPT4Point_Cap3D_Eval_Processor

)

from lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    "GPT4Point_Cap3D_Train_Processor",
    "GPT4Point_Cap3D_Eval_Processor",
    "BlipCaptionProcessor"
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("gpt4point_cap3d_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
