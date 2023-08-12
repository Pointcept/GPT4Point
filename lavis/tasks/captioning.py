"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.highligt_ids = ['2f5515b4e59545c8b1f0ba1365ed35af', '7a3f2d273f354b29b31f247beb62d973', '7eee9870d8664dee9966228052d249ab', '8c7e0cb5e5ae4789865294e53aaaee3e',
                             '9ea93fce9b9746f5bd08ce115591aa65', '442fca07d5394a0e922b8c6273757e66', '727a0c2c5ee74aa8adffe1c8502ed225', '1239c5354dc6463ba4142f7fbd921e07',
                             'a1846945780a4a7fb0c7c0cfb1dfebcd', 'bde005021ed143288e74898b0f7f9f51', 'c4c09479570943e2845fbd4c6a450568', 'eb399e07a5224790ac6a90b8e0922ce9',
                             'fa4bd59f2bd14cc583200f402a10b27c']

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

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        
        img_ids = samples["ann_id"]
        text_inputs = samples["text_input"]
        for text_input, caption, img_id in zip(text_inputs, captions, img_ids):
            results.append({"2d_caption": text_input, "caption": caption, "image_id": img_id})
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        hi_res = []
        for res in val_result:
            if res['image_id'] in self.highligt_ids:
                hi_res.append(res)
        eval_result_file = self.save_result(
            result=hi_res,
            result_dir=registry.get_path("result_dir"),
            filename="{}_example_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        # MOD no evaluation
        # if self.report_metric:
        #     metrics = self._report_metrics(
        #         eval_result_file=eval_result_file, split_name=split_name
        #     )
        # else:
        #     metrics = {"agg_metrics": 0.0}

        return {"agg_metrics": 0.0}

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
