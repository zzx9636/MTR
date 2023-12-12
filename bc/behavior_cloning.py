from typing import Any

import torch
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader

from mtr.datasets.waymo.waymo_dataset_bc import WaymoDatasetBC as WaymoDataset
from mtr.config import cfg, cfg_from_yaml_file

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from mtr.models.context_encoder.mtr_encoder import MTREncoder
from mtr.models.motion_decoder.bc_decoder import BCDecoder


from easydict import EasyDict


class BehaviorCloning(pl.LightningModule):
    def __init__(self, 
                 cfg: EasyDict, 
                 pretrained_model: dict = None, 
                 freeze_encoder: bool = False
                ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['logger', 'pretrained_model', 'freeze_pretrained'])
        # Save the config
        self.cfg = cfg
        self.opt_cfg = cfg.OPTIMIZATION
        self.epoch = 0
        self.freeze_encoder = freeze_encoder
        
        self.encoder = MTREncoder(
            cfg.MODEL.CONTEXT_ENCODER
            )
        self.decoder = BCDecoder(
            self.encoder.num_out_channels,
            cfg.MODEL.MOTION_DECODER
        )
                
        if pretrained_model is not None:
            self.encoder.load_model(pretrained_model)
        
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        param2opt = self.decoder.parameters()
        if not self.freeze_encoder:
            param2opt.extend(self.encoder.parameters())
                 

        optimizer = torch.optim.AdamW(
            param2opt, 
            lr=self.opt_cfg.LR,
            weight_decay=self.opt_cfg.get('WEIGHT_DECAY', 0))

        scheduler = lr_sched.StepLR(
            optimizer, 
            step_size=self.opt_cfg.LR_STEP,
            gamma=self.opt_cfg.LR_GAMMA
        )
        
        if scheduler is not None:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
        
    def training_step(self, batch, batch_idx):
        encoder_output = self.encoder(batch, retain_input=True)
        decoder_output = self.decoder(encoder_output)
        loss, tb_dict = self.decoder.get_loss(decoder_output, 'train/')
         
        log_dict = {f'train/{k}': v for k, v in tb_dict.items()}
        self.log_dict(log_dict, on_step=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        encoder_output = self.encoder(batch, retain_input=True)
        decoder_output = self.decoder(encoder_output)
        _, tb_dict = self.decoder.get_loss(decoder_output, 'val/')
         
        log_dict = {f'train/{k}': v for k, v in tb_dict.items()}
        self.log_dict(log_dict, on_step=True, prog_bar=True, logger=True)