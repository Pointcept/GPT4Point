
import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


from lavis.datasets.transforms.transforms_point import pc_norm, random_point_dropout, random_scale_point_cloud, shift_point_cloud, rotate_perturbation_point_cloud, rotate_point_cloud


@registry.register_processor("pointblip_objaverse_point_train")
class PointBlipObjaPointTrainProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.pc_form = pc_norm
        self.random_point_dropout = random_point_dropout
        self.random_scale_point_cloud = random_scale_point_cloud
        self.shift_point_cloud = shift_point_cloud
        self.rotate_perturbation_point_cloud = rotate_perturbation_point_cloud
        self.rotate_point_cloud = rotate_point_cloud
        # point = point.squeeze()                                                           # (1, 8192, 3) -> (8192, 3)

    def __call__(self, item):
        item = self.pc_form(item)
        item = self.random_point_dropout(item[None, ...])                                    # (1, 8192, 3) -> (1, 8192, 3)
        item = self.random_scale_point_cloud(item)
        item = self.shift_point_cloud(item)
        item = self.rotate_perturbation_point_cloud(item)
        item = self.rotate_point_cloud(item)
        item = item.squeeze()                                                           # (1, 8192, 3) -> (8192, 3)

        return item


@registry.register_processor("pointblip_objaverse_point_eval")
class PointBlipObjaPointEvalProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.pc_form = pc_norm

    def __call__(self, item):
        item = self.pc_form(item)
        return item