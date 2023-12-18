import torch
import torch.optim.lr_scheduler as lr_sched
import lightning.pytorch as pl

from mtr.models.context_encoder.mtr_encoder import MTREncoder

from mtr.models.motion_decoder.bc_decoder import BCDecoder as BCDecoderBicycle
from mtr.models.motion_decoder.bc_decoder_delta import BCDecoder as BCDecoderDelta
from mtr.models.motion_decoder.bc_decoder_discrete import BCDecoder as BCDecoderDiscrete

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
        
        if cfg.MODEL.MOTION_DECODER.TYPE == 'bicycle':
            BCDecoder = BCDecoderBicycle
        elif cfg.MODEL.MOTION_DECODER.TYPE == 'delta':
            BCDecoder = BCDecoderDelta
        elif cfg.MODEL.MOTION_DECODER.TYPE == 'discrete':
            BCDecoder = BCDecoderDiscrete
        else:
            raise ValueError(f'Unknown motion decoder type: {cfg.MODEL.MOTION_DECODER.TYPE}')
        
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
        
        def lr_foo(step, warmup_step, step_size, gamma):

            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step)/warmup_step
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma**n

            return lr_scale

        scheduler = lr_sched.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_foo(
                step, 
                self.opt_cfg.LR_WARMUP, 
                self.opt_cfg.LR_STEP,
                self.opt_cfg.LR_GAMMA
            )
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