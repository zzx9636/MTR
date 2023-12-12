from typing import Any

import torch
from torch.utils.data import DataLoader

from mtr.config import cfg, cfg_from_yaml_file

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from waymax import config as waymax_config

from bc.behavior_cloning import BehaviorCloning
from bc.bc_dataset import BCDatasetBuffer

    
# main function
def train(cfg_file, 
        pretrained_encoder: dict = None, 
        freeze_pretrained: bool = True
    ):

    # Load the config
    cfg_from_yaml_file(cfg_file, cfg)
    
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        
    # Construct Dataset
    train_dataset = BCDatasetBuffer(
        waymax_config.DatasetConfig(
            path=cfg.DATA.TRAIN_PATH,
            max_num_rg_points=30000,
            data_format=waymax_config.DataFormat.TFRECORD,
            max_num_objects=32,
        ),
        buffer_size=1000,
    )
    
    val_dataset = BCDatasetBuffer(
        waymax_config.DatasetConfig(
            path=cfg.DATA.VAL_PATH,
            max_num_rg_points=30000,
            data_format=waymax_config.DataFormat.TFRECORD,
            max_num_objects=32,
        ),
        buffer_size=500,
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        num_workers=0,
        shuffle=False, 
        collate_fn=train_dataset.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True, 
        num_workers=0,
        shuffle=False, 
        collate_fn=val_dataset.collate_fn,
    )
    
    model = BehaviorCloning(cfg, pretrained_encoder, freeze_pretrained)
    
    logger = WandbLogger(project='MTR_BC_CTRL', entity='zzx9636', log_model=True)
    logger.watch(model, log_freq=500)
    
    # logger = None
    num_decoder = cfg.MODEL.MOTION_DECODER.NUM_DECODER_LAYERS
    freeze_str = 'freeze' if freeze_pretrained else 'unfreeze'
    
    trainer = pl.Trainer(
        max_steps=cfg.OPTIMIZATION.MAX_STEPS,
        val_check_interval=cfg.OPTIMIZATION.VAL_CHECK_INTERVAL,
        limit_val_batches = cfg.OPTIMIZATION.LIMIT_VAL_BATCHES,
        accelerator="gpu",
        enable_progress_bar=True, 
        logger=logger, 
        detect_anomaly=True,
        gradient_clip_val=0.5, 
        gradient_clip_algorithm="value",
        callbacks=[
            ModelCheckpoint(
                dirpath = f'output/bc_ctrl_{num_decoder}_{freeze_str}',
                save_top_k=10,
                monitor='val/loss_total', 
                every_n_train_steps = cfg.OPTIMIZATION.VAL_CHECK_INTERVAL,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval='step')
        ]
    )
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == '__main__':
    encoder_state_dict = torch.load('model/checkpoint_epoch_30.pth')['model_state']
    train(
        'tools/cfgs/waymo/bc_atten_ctrl.yaml',
        encoder_state_dict,
        True,
    )

         