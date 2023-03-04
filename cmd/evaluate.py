import os
import json
import logging
from typing import Dict, List, Tuple, Sequence, cast

import torch
import numpy as np
from monai.data.dataloader import DataLoader
from monai.data.utils import decollate_batch  # type: ignore
from monai.utils.enums import MetricReduction
from monai.metrics.meandice import DiceMetric

from base.util import setup_logging
from dataset import option as dataset_option
from dataset.loader import create_data_loader, LABEL_KEY, PREDICTION_KEY

from .argument import EvaluateArgument

T = Sequence[torch.Tensor]


def channelwise_dices(
    truth: torch.Tensor,
    preds: torch.Tensor,
) -> Tuple[List[float], List[int]]:
    """Calculate batch-averaged channel-wise Dice metrics.

    Args:
        truth (torch.Tensor): The grond truth of shape (B, C, *dims).
        preds (torch.Tensor): The prediction of shape (B, C, *dims).

    Returns:
        Tuple[List[float], List[int]]:
            0: A list of length `C` of metric for each channel, and will be 0 if the
                metric is invalid, e.g. involving division by zero.
            1: A list of length `C` of the number of batches contribtued to the average,
                i.e. those bathes with positive denominators when calculating Dices.
    """
    truth_list = cast(T, decollate_batch(truth))
    preds_list = cast(T, decollate_batch(preds))

    metric_func = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    metric_func(y_pred=preds_list, y=truth_list)
    metric, not_nans = cast(T, metric_func.aggregate())

    return metric.cpu().numpy(), not_nans.cpu().numpy()


def _start(truth_loader: DataLoader):
    total_nums = np.zeros(3)
    total_dices = np.zeros(3)
    for idx, batch in enumerate(truth_loader, 1):
        fpaths = batch[f"{LABEL_KEY}_meta_dict"]["filename_or_obj"]
        fnames = list(map(os.path.basename, fpaths))

        truth = batch[LABEL_KEY]
        preds = batch[PREDICTION_KEY]

        dices, nums = channelwise_dices(truth, preds)

        [dice_tc, dice_wt, dice_et] = dices
        logging.info(
            "Batch %s: WT(1+2+4) Dice: %.04f, TC(1+4) Dice: %.04f, ET(4) Dice: %.04f (%d/%d)",
            fnames,
            dice_wt,
            dice_tc,
            dice_et,
            idx,
            len(truth_loader),
        )

        total_nums += nums
        total_dices += dices

    avg = total_dices / total_nums
    [dice_tc, dice_wt, dice_et] = avg
    logging.info(
        "Average: WT(1+2+4) Dice: %.04f, TC(1+4) Dice: %.04f, ET(4) Dice: %.04f",
        dice_wt,
        dice_tc,
        dice_et,
    )


def _main():
    # parse arguments
    args = EvaluateArgument().parse_args()
    setup_logging(level=args.log_level)

    # build a dict to look up predictions
    pdir = args.prediction_dir
    pred_lookup: Dict[str, str] = {}
    for case_name in os.listdir(pdir):
        fpath = os.path.join(pdir, case_name, f"{case_name}_pred.nii.gz")
        if os.path.isfile(fpath):
            pred_lookup[case_name] = fpath

    # create a loader to iterate ground truth
    with open(args.fold_map, encoding="utf-8") as file:
        fold_map = json.load(file)

    data_loader = create_data_loader(
        args.data_root,
        fold_map,
        args.folds,
        dataset_option.with_batch_size(1),
        dataset_option.with_image_included(False),
        dataset_option.with_label_included(True),
        dataset_option.with_prediction_path_lookup(lambda key: pred_lookup[key]),
    )

    _start(truth_loader=data_loader)


if __name__ == "__main__":
    _main()
