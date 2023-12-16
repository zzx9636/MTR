import torch
import torch.optim.lr_scheduler as lr_sched
import lightning.pytorch as pl

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
        else:
            # make all parameters in encoder not trainable
            for param in self.encoder.parameters():
                param.requires_grad = False
            
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
        self.log_dict(tb_dict, on_step=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        encoder_output = self.encoder(batch, retain_input=True)
        decoder_output = self.decoder(encoder_output)
        _, tb_dict = self.decoder.get_loss(decoder_output, 'val/')
        self.log_dict(tb_dict, on_step=True, prog_bar=True, logger=True)