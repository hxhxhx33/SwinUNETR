from model.network import SwinUNETR
from model.swin.swin import Swin, PatchEmbedding

from .argument import ModelArgument


def create_model(args: ModelArgument) -> SwinUNETR:
    """Create the model.

    Args:
        args (ModelArgument): Arguments specifying model configurations.

    Returns:
        SwinUNETR: The built model.
    """
    opt = SwinUNETR.Options(
        output_channel=args.output_channel,
        swin_options=Swin.Options(
            window_size=args.window_size,
            stage_depths=args.swin_stage_depths,
            stage_num_heads=args.swin_stage_num_heads,
            block_mlp_ratio=args.mlp_ratio,
            embed_options=PatchEmbedding.Options(
                num_dim=args.num_dim,
                patch_size=args.patch_size,
                input_channel=args.input_channel,
                output_channel=args.hidden_channel,
            ),
        ),
    )
    return SwinUNETR(opt)
