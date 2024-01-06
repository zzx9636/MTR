import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# set tf to cpu only
tf.config.set_visible_devices([], 'GPU')
import jax
jax.config.update('jax_platform_name', 'cpu')


from typing import Any
import torch
from torch.utils.data import DataLoader
from mtr.config import cfg, cfg_from_yaml_file
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from bc.behavior_cloning import BehaviorCloning
from bc.bc_dataset import BCDataset

    
# main function
def train(
    cfg_file, 
    pretrained_encoder: dict = None, 
    freeze_pretrained: bool = True,
    sample_method: str = 'uniform',
):
    torch.set_float32_matmul_precision('high')

    # Load the config
    cfg_from_yaml_file(cfg_file, cfg)
    
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        
    # Construct Dataset
    train_dataset = BCDataset(
        cfg.DATA.TRAIN_PATH,
        sample_method = sample_method,
    )
    
    val_dataset = BCDataset(
        cfg.DATA.VAL_PATH,
        sample_method = sample_method,
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        num_workers=16,
        shuffle=False, 
        collate_fn=train_dataset.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True, 
        num_workers=16,
        shuffle=False, 
        collate_fn=val_dataset.collate_fn,
    )
    # logger = None
    decoder_type = cfg.MODEL.MOTION_DECODER.TYPE
    num_decoder = cfg.MODEL.MOTION_DECODER.NUM_DECODER_LAYERS
    freeze_str = 'freeze' if freeze_pretrained else 'unfreeze'
    
    # model = BehaviorCloning(cfg, pretrained_encoder, freeze_pretrained)
    model = BehaviorCloning.load_from_checkpoint(
        'output/bc_discrete_4_freeze_2/epoch=0-step=920000.ckpt',
    )
    
    logger = WandbLogger(project=f'MTR_BC_{decoder_type}', entity='zzx9636', log_model=True)
    logger.watch(model, log_freq=500)
    
    trainer = pl.Trainer(
        max_steps=cfg.OPTIMIZATION.MAX_STEPS,
        # val_check_interval=cfg.OPTIMIZATION.VAL_CHECK_INTERVAL,
        # limit_val_batches = cfg.OPTIMIZATION.LIMIT_VAL_BATCHES,
        accelerator="gpu",
        enable_progress_bar=True, 
        logger=logger, 
        detect_anomaly=False,
        gradient_clip_val=0.5, 
        gradient_clip_algorithm="value",
        callbacks=[
            # ModelCheckpoint(
            #     dirpath = f'output/bc_{decoder_type}_{num_decoder}_{freeze_str}',
            #     save_top_k=100,
            #     save_weights_only = True,
            #     monitor='val/loss_total', 
            #     every_n_epochs = 1,
            #     save_on_train_epoch_end=False,
            # ),
            ModelCheckpoint(
                dirpath = f'output/bc_{decoder_type}_{num_decoder}_{freeze_str}',
                save_top_k=100,
                save_weights_only = True,
                monitor='train/loss_total', 
                every_n_train_steps = 10000,
                # save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval='step')
        ]
    )
    trainer.fit(model, train_loader)#, val_loader)
    
if __name__ == '__main__':
    encoder_state_dict = torch.load('model/checkpoint_epoch_30.pth')['model_state']
    train(
        # 'tools/cfgs/waymo/bc_atten_ctrl_full.yaml',
        'tools/cfgs/waymo/bc_atten_discrete.yaml',
        encoder_state_dict,
        True,
        'raw'
    )

         