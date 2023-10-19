from typing import Any

import torch
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader

from mtr.datasets.waymo.waymo_dataset_bc import WaymoDatasetBC as WaymoDataset
from mtr.config import cfg, cfg_from_yaml_file
from mtr.models import model as model_utils

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger


from easydict import EasyDict

class PrintLogger():
    @staticmethod
    def info(msg: str):
        print(msg)

class MTR_Lightning(pl.LightningModule):
    def __init__(self, cfg: EasyDict, logger = None, 
                 pretrained_model: str = None, freeze_pretrained: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['logger', 'pretrained_model', 'freeze_pretrained'])
        # Save the config
        self.cfg = cfg
        self.opt_cfg = cfg.OPTIMIZATION
        self.epoch = 0
        self.model = model_utils.MotionTransformer(config = cfg.MODEL)
        
        if pretrained_model is not None:
            self.model.load_params_from_file(
                filename=pretrained_model,
                logger=logger,
                freeze_pretrained = freeze_pretrained,
                keys_to_ignore = 'motion_decoder'
            )
        
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        
        param2opt = [param for param in self.model.parameters() if param.requires_grad]
        # print('motion_decoder.query' in param2opt)
        # print(param2opt)
        if self.opt_cfg.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(
                param2opt,
                lr=self.opt_cfg.LR, 
                weight_decay=self.opt_cfg.get('WEIGHT_DECAY', 0)
            )
        elif self.opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(
                param2opt, 
                lr=self.opt_cfg.LR,
                weight_decay=self.opt_cfg.get('WEIGHT_DECAY', 0))
        else:
            assert False
            
        # Set up the learning rate scheduler
        
        last_epoch = self.epoch-1
        if self.opt_cfg.get('SCHEDULER', None) == 'cosine':
            scheduler = lr_sched.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=2 * self.opt_cfg.DATA_LOADER_SIZE,
                T_mult=1,
                eta_min=max(1e-2 * self.opt_cfg.LR, 1e-6),
                last_epoch=-1,
            )
        elif self.opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
            total_iters_each_epoch = self.opt_cfg.DATA_LOADER_SIZE // self.opt_cfg.BATCH_SIZE_PER_GPU
            
            decay_steps = [x * total_iters_each_epoch for x in self.opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20])]
            def lr_lbmd(cur_epoch):
                cur_decay = 1
                for decay_step in decay_steps:
                    if cur_epoch >= decay_step:
                        cur_decay = cur_decay * self.opt_cfg.LR_DECAY
                return max(cur_decay, self.opt_cfg.LR_CLIP / self.opt_cfg.LR)
            
            scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
        elif self.opt_cfg.get('SCHEDULER', None) == 'linearLR':
            total_iters = total_iters_each_epoch * self.opt_cfg.NUM_EPOCHS
            scheduler = lr_sched.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.opt_cfg.LR_CLIP / self.opt_cfg.LR,
                total_iters=total_iters,
                last_epoch=last_epoch,
            )
        else:
            scheduler = None
        
        if scheduler is not None:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
        
    def training_step(self, batch, batch_idx):
        batch_size = len(batch['input_dict']['track_index_to_predict'])
        output = self.model(batch)
        loss = output[0]
        tb_dict = output[1]
        
        log_dict = {f'train/{k}': v for k, v in tb_dict.items()}
        self.log_dict(log_dict, on_step=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch['input_dict']['track_index_to_predict'])
        output = self.model(batch)
        tb_dict = output[1]
        log_dict = {f'val/{k}': v for k, v in tb_dict.items()}
        self.log_dict(log_dict, on_step=False, prog_bar=True, logger=True, batch_size=batch_size)
        
    def forward(self, batch, get_loss: bool = False):
        return self.model(batch, get_loss)
    
    def sample(self, batch):
        self.eval()
        with torch.no_grad():
            batch_dict = self.model(batch, get_loss=False)
            pred_scores = batch_dict['pred_scores']
            pred_ctrls = batch_dict['pred_ctrls']
            mode, mix, gmm = self.model.motion_decoder.build_gmm_distribution(pred_ctrls, pred_scores)
            # batch_size = pred_scores.shape[0]
            sample = gmm.sample()#.cpu().numpy()
            sample = (sample * self.model.motion_decoder.output_std + self.model.motion_decoder.output_mean).cpu().numpy()
        return mode, mix, gmm, sample
        
        
        
# main function
def train(cfg_file, pretrained_model, freeze_pretrained):
    logger = PrintLogger()
    # Load the config
    cfg_from_yaml_file(cfg_file, cfg)
    
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    epochs = cfg.OPTIMIZATION.NUM_EPOCHS
    
    # Construct Dataset
    train_dataset = WaymoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        training=True,
        logger=logger, 
    )
    
    val_dataset = WaymoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        training=False,
        logger=logger, 
    )
    
    cfg.OPTIMIZATION.DATA_LOADER_SIZE = len(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, num_workers=8,
        shuffle=True, collate_fn=train_dataset.collate_batch,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8,
        shuffle=False, collate_fn=val_dataset.collate_batch,
        drop_last=False
    )
    
    model = MTR_Lightning(cfg, logger, pretrained_model, freeze_pretrained)
    
    logger = WandbLogger(project='MTR_BC', entity='zzx9636', log_model = True)
    logger.watch(model, log_freq=100)
    # logger = None
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        enable_progress_bar=True, 
        logger=logger, 
        detect_anomaly=True,
        gradient_clip_val=0.5, gradient_clip_algorithm="value",
        callbacks=[
            ModelCheckpoint(
                dirpath = 'output/bc',
                save_top_k=10,
                save_last=True,
                monitor='val/loss_total', 
                save_on_train_epoch_end=True
            ),
            LearningRateMonitor(logging_interval='step')
        ]
    )
    
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == '__main__':
    train(
        'tools/cfgs/waymo/bc+10_percent_data.yaml',
        'model/checkpoint_epoch_30.pth',
        False
    )

         