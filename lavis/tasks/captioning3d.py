"""
 Copyright (c) 2022, PointBLIP.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

from lavis.datasets.datasets.cap3d_captioning3d_dataset import cap3d_captioning3d_eval

@registry.register_task("captioning3d")
class Captioning3dTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate,report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        
        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples, return_before_evaluation=None):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        pcd_ids = samples["pcd_id"]
        for caption, pcd_id in zip(captions, pcd_ids):
            results.append({"caption": caption, "pcd_id": pcd_id})

        return results


    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        if 'final' in kwargs:
            eval_result_file = kwargs['eval_result_file']
        else:
            eval_result_file = self.save_result(
                result=val_result,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}".format(split_name, epoch),
                remove_duplicate="pcd_id",
            )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        ptstext_eval = cap3d_captioning3d_eval(eval_result_file, split_name)
        agg_metrics = ptstext_eval.eval["CIDEr"] + ptstext_eval.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in ptstext_eval.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        ptstext_results = {k: v for k, v in ptstext_eval.eval.items()}
        ptstext_results["agg_metrics"] = agg_metrics

        return ptstext_results