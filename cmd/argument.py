from typing import List

from base.argument import (
    BaseArgument,
    CommonArgument,
    DatasetArgument as BaseDatasetArgument,
    TrainArgument as BaseTrainArgument,
    PredictArgument as BasePredictArgument,
)


class DatasetArgument(BaseDatasetArgument):
    """Commend line arguments for data set configurations."""

    no_random_rotate: bool = False
    """if not to randomly rotate the input"""

    no_augment: bool = False
    """if not to augment"""

    no_crop_random_center: bool = False
    """if not to randomly pick center when cropping"""


class ModelArgument(BaseArgument):
    """Commend line arguments for model."""

    num_dim: int
    """number of dimensionality"""

    patch_size: int
    """patch size"""

    window_size: int
    """window size"""

    swin_stage_depths: List[int]
    """depths for each swin stage"""

    swin_stage_num_heads: List[int]
    """number of attention heads for each swin stage"""

    input_channel: int
    """number of input channel"""

    hidden_channel: int
    """number of hidden channel"""

    output_channel: int
    """number of output channel"""

    mlp_ratio: int
    """mlp ratio"""


class TrainArgument(DatasetArgument, ModelArgument, BaseTrainArgument):
    """Commend line arguments for training."""

    spatial_size: int
    """spatial dimensions to which the input data will be padded thus must be larger
    than all expected input data size"""


class PredictArgument(ModelArgument, BasePredictArgument):
    """Commend line arguments for prediction."""

    spatial_size: int
    """spatial dimensions to which the input data will be padded thus must be larger
    than all expected input data size"""

    sw_batch_size: int = 1
    """batch size for sliding window inference"""

    sw_overlap_ratio: float = 0.6
    """overlap ratio for sliding window inference"""


class EvaluateArgument(CommonArgument):
    """Commend line arguments for evaluate."""

    prediction_dir: str
    """directory in which to-be-evaluated predictions are saved"""
