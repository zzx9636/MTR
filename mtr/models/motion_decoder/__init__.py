# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


from .mtr_decoder import MTRDecoder
from .bc_decoder import BCDecoder

__all__ = {
    'MTRDecoder': MTRDecoder,
    'BCDecoder': BCDecoder,
}


def build_motion_decoder(in_channels, config):
    model = __all__[config.NAME](
        in_channels=in_channels,
        config=config
    )

    return model
