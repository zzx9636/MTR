# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_encoder import build_context_encoder
from .motion_decoder import build_motion_decoder


class MotionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER)
        self.motion_decoder = build_motion_decoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER
        )

    def forward(self, batch_dict, get_loss=True):
        batch_dict = self.context_encoder(batch_dict)
        batch_dict = self.motion_decoder(batch_dict)

        if get_loss:
            loss, tb_dict = self.get_loss()
            return loss, tb_dict, batch_dict
        else:
            return batch_dict


    def get_loss(self):
        loss, tb_dict = self.motion_decoder.get_loss()

        return loss, tb_dict

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if logger is not None:
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if logger is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        # logger.info('==> Done')
        if logger is not None:
            logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger = None, 
                              to_cpu=False, freeze_pretrained = False, 
                              keys_to_ignore = None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if logger is not None:
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        else:
            print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
            
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None and logger is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)
        else:
            print('==> Checkpoint trained from version: %s' % version)
        if logger is not None:
            logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        else:
            print(f'The number of disk ckpt keys: {len(model_state_disk)}')
        
        model_state = self.state_dict()
        model_state_disk_filter = {}
        
        for key, val in model_state_disk.items():
            if keys_to_ignore is not None and keys_to_ignore in key:
                continue
            elif key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)
        
        if freeze_pretrained:
            for name, param in self.named_parameters():
                if name in model_state_disk:
                    param.requires_grad = False
                    
        if logger is not None:
            # logger.info(f'Missing keys: {missing_keys}')
            logger.info(f'The number of missing keys: {len(missing_keys)}')
            logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
            logger.info('==> Done (total keys %d)' % (len(model_state)))
        else:
            # print(f'Missing keys: {missing_keys}')
            print(f'The number of missing keys: {len(missing_keys)}')
            print(f'The number of unexpected keys: {len(unexpected_keys)}')
            print('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch


