# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtr.models.utils.transformer import transformer_decoder_layer
from mtr.models.utils.transformer import position_encoding_utils
from mtr.models.utils import common_layers
from mtr.utils import common_utils, loss_utils, motion_utils
from mtr.config import cfg
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical

class SimpleBCDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layer = self.model_cfg.NUM_DECODER_LAYERS
        
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
        mlp = common_layers.build_mlps(c_in=self.d_model,
                                        mlp_channels=[self.d_model, self.d_model],
                                        without_norm=True,
                                        ret_before_act=False)
        
        self.mlp_list = nn.ModuleList([copy.deepcopy(mlp) for _ in range(self.num_decoder_layer)])
        
  

        self.prediction_head = common_layers.build_mlps(c_in=self.d_model, 
                                                        mlp_channels=[self.d_model, self.d_model, 7*self.num_motion_modes], 
                                                        without_norm=True,
                                                        ret_before_act=True)
        
        self.register_buffer('output_mean', torch.tensor([7.26195561e-01, 1.52434988e-03, 7.25015970e-04]))
        self.register_buffer('output_std', torch.tensor([0.54773382, 0.02357974, 0.04607823]))
        
        self.forward_ret_dict = {}

    def build_mode_distribution(self, pred_ctrl, log_std_range=(-5.0, 2.0), rho_limit=0.4):
        independent = pred_ctrl.shape[-1] == 6
    
        mean = pred_ctrl[..., 0:3] # (num_center_objects, num_query, 3)
                
        log_std = torch.clip(pred_ctrl[..., 3:6], min=log_std_range[0], max=log_std_range[1])
        std = torch.exp(log_std)
        
        std1 = std[..., 0]
        std2 = std[..., 1]
        std3 = std[..., 2]
        if independent:
            rho1 = rho2 = rho3 = torch.zeros_like(std1)
        else:
            rho1 = torch.clip(pred_ctrl[..., 6], min=-rho_limit, max=rho_limit) # 1&2
            rho2 = torch.clip(pred_ctrl[..., 7], min=-rho_limit, max=rho_limit) # 1&3
            rho3 = torch.clip(pred_ctrl[..., 8], min=-rho_limit, max=rho_limit) # 2&3 
            
        covariance = torch.stack([
            torch.stack([std1**2, rho1*std1*std2, rho2*std1*std3], dim=-1),
            torch.stack([rho1*std1*std2, std2**2, rho3*std2*std3], dim=-1),
            torch.stack([rho2*std1*std3, rho3*std2*std3, std3**2], dim=-1),
        ], dim=-1) # (num_center_objects, num_query, 3, 3)
        mode = MultivariateNormal(mean, covariance_matrix=covariance)
      
        return mode
    
    
    def build_gmm_distribution(self, pred_ctrl, pred_ctrl_score, log_std_range=(-5.0, 2.0), rho_limit=0.4):
        mode = self.build_mode_distribution(pred_ctrl, log_std_range, rho_limit)
        mix = Categorical(logits=pred_ctrl_score)
        gmm = MixtureSameFamily(mix, mode)
        return mode, mix, gmm
    
    def get_loss(self, tb_pre_tag=''):
        return self.get_loss_best(tb_pre_tag)
        # return self.get_loss_gmm(tb_pre_tag)

    def get_loss_best(self, tb_pre_tag=''):
        tb_dict = {}
        
        center_gt = self.forward_ret_dict['center_gt'][...,None,:3].cuda()
        # normalize the gt
        center_gt = (center_gt - self.output_mean) / self.output_std
        
        pred_ctrls = self.forward_ret_dict['pred_ctrls']
        pred_scores = self.forward_ret_dict['pred_scores']
        
        # Get mode for all
        mode_all = self.build_mode_distribution(pred_ctrls) # [batch size]
        nll_loss_all = -mode_all.log_prob(center_gt)
        nll_loss_best, best_idx = nll_loss_all.min(dim=-1)

        cls_loss = F.cross_entropy(input  = pred_scores, target= best_idx, reduction='none')

        total_loss = (nll_loss_best + cls_loss).mean()
        tb_dict[f'{tb_pre_tag}loss_nll'] = nll_loss_best.mean().item()
        tb_dict[f'{tb_pre_tag}loss_cls'] = cls_loss.mean().item()
        tb_dict[f'{tb_pre_tag}loss_total'] = total_loss.item()
        
        # record mean and std
        tb_dict[f'{tb_pre_tag}input_mean'] = self.forward_ret_dict['input_mean'].item()
        tb_dict[f'{tb_pre_tag}input_std'] = self.forward_ret_dict['input_std'].item()
        tb_dict[f'{tb_pre_tag}feature_mean'] = self.forward_ret_dict['feature_mean'].item()
        tb_dict[f'{tb_pre_tag}feature_std'] = self.forward_ret_dict['feature_std'].item()
        tb_dict[f'{tb_pre_tag}pred_mean'] = self.forward_ret_dict['pred_mean'].item()
        tb_dict[f'{tb_pre_tag}pred_std'] = self.forward_ret_dict['pred_std'].item()
        tb_dict[f'{tb_pre_tag}score_mean'] = self.forward_ret_dict['score_mean'].item()
        tb_dict[f'{tb_pre_tag}score_std'] = self.forward_ret_dict['score_std'].item()
            
        return total_loss, tb_dict
    
    def get_loss_gmm(self, tb_pre_tag=''):
        tb_dict = {}
        
        center_gt = self.forward_ret_dict['center_gt'][...,:3].cuda()
        # normalize the gt
        center_gt = (center_gt - self.output_mean) / self.output_std
        
        pred_ctrls = self.forward_ret_dict['pred_ctrls']
        pred_scores = self.forward_ret_dict['pred_scores']
        
        # Get mode for all
        _, _, gmm = self.build_gmm_distribution(pred_ctrls, pred_scores) # [batch size]
        nll_loss = -gmm.log_prob(center_gt)

        # cls_loss = F.cross_entropy(input  = pred_scores, target= best_idx, reduction='none')

        total_loss = nll_loss.mean()
        tb_dict[f'{tb_pre_tag}loss_nll'] = nll_loss.mean().item()
        # tb_dict[f'{tb_pre_tag}loss_cls'] = cls_loss.mean().item()
        tb_dict[f'{tb_pre_tag}loss_total'] = total_loss.item()
        
        # record mean and std
        tb_dict[f'{tb_pre_tag}input_mean'] = self.forward_ret_dict['input_mean'].item()
        tb_dict[f'{tb_pre_tag}input_std'] = self.forward_ret_dict['input_std'].item()
        tb_dict[f'{tb_pre_tag}feature_mean'] = self.forward_ret_dict['feature_mean'].item()
        tb_dict[f'{tb_pre_tag}feature_std'] = self.forward_ret_dict['feature_std'].item()
        tb_dict[f'{tb_pre_tag}pred_mean'] = self.forward_ret_dict['pred_mean'].item()
        tb_dict[f'{tb_pre_tag}pred_std'] = self.forward_ret_dict['pred_std'].item()
        tb_dict[f'{tb_pre_tag}score_mean'] = self.forward_ret_dict['score_mean'].item()
        tb_dict[f'{tb_pre_tag}score_std'] = self.forward_ret_dict['score_std'].item()
        return total_loss, tb_dict
   
    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects = center_objects_feature.shape[0]
        
        self.forward_ret_dict['input_mean'] = center_objects_feature.mean()
        self.forward_ret_dict['input_std'] = center_objects_feature.std()

        center_objects_feature = self.input_proj(center_objects_feature).view(-1, self.d_model)
        
        self.forward_ret_dict['feature_mean'] = center_objects_feature.mean()
        self.forward_ret_dict['feature_std'] = center_objects_feature.std()
        
        for mlp in self.mlp_list:
            # residual connection
            center_objects_feature = mlp(center_objects_feature) + center_objects_feature
        
        prediction = self.prediction_head(center_objects_feature)
        
        prediction = prediction.view(num_center_objects, self.num_motion_modes, -1)
        
        # Generate 9D Control
        # [dx, dy, dtheta, sigma_x, signa_y, sigma_theta, rho_xy, rho_xtheta, rho_ytheta]
        pred_scores = prediction[..., -1]
        pred_ctrls = prediction[..., :-1]
        
        self.forward_ret_dict['pred_mean'] = pred_ctrls.mean()
        self.forward_ret_dict['pred_std'] = pred_ctrls.std()
        self.forward_ret_dict['score_mean'] = pred_scores.mean()
        self.forward_ret_dict['score_std'] = pred_scores.std()
        
        if 'center_gt' in input_dict:
            self.forward_ret_dict['pred_scores'] = pred_scores
            self.forward_ret_dict['pred_ctrls'] = pred_ctrls
            self.forward_ret_dict['center_gt'] = input_dict['center_gt']
            # Otherwise, it is in the inference mode
            
        batch_dict['pred_scores'] = pred_scores
        batch_dict['pred_ctrls'] = pred_ctrls

        return batch_dict
