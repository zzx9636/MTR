# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtr.models.utils.transformer.transformer_decoder_layer import TransformerDecoder
from mtr.models.utils.transformer import position_encoding_utils
from mtr.models.utils.common_layers import ResidualMLP
# from mtr.utils import common_utils, loss_utils, motion_utils
from mtr.config import cfg
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
import numpy as np
                   
class BCDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.d_model = self.model_cfg.D_MODEL
        self.n_head = self.model_cfg.NUM_ATTN_HEAD
        self.dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1)
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS
        self.loss_mode = self.model_cfg.get('LOSS_MODE', 'best')
        self.use_bicycle_model = self.model_cfg.get('USE_BICYCLE_MODEL', False)
        
        # Build the query
        self.query = nn.Parameter(torch.randn(self.num_motion_modes, self.d_model), requires_grad=True)
        
        # Project the input to a higher dimension
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
        self.in_proj_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
        self.in_proj_map = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
        # Query Fusion
        self.pre_query_fusion_layer = nn.Sequential(
            nn.Linear(self.d_model*2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        ) 
        
        if self.num_decoder_layers > 0:
            # Attention        
            self.obj_atten_layers = nn.ModuleList([
                TransformerDecoder(
                    d_model=self.d_model, n_head=self.n_head,
                    dim_feedforward=self.d_model*4, dropout=self.dropout, 
                    with_self_atten=False, normalize_before=True
                ) for _ in range(self.num_decoder_layers)
                ])
            
            self.map_atten_layers = nn.ModuleList([TransformerDecoder(
                    d_model=self.d_model, n_head=self.n_head,
                    dim_feedforward=self.d_model*4, dropout=self.dropout,
                    with_self_atten=False, normalize_before=True
                ) for _ in range(self.num_decoder_layers)])
            
           
            self.query_fusion_layers = nn.ModuleList([nn.Sequential(
                nn.Linear(self.d_model*3, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            ) for _ in range(self.num_decoder_layers)]) 
            
        
        # Prediction Head
        if self.use_bicycle_model:
            output_dim = 9
        else:
            output_dim = 10
            
        self.prediction_layers = nn.ModuleList([ResidualMLP(
                c_in = self.d_model,
                c_out = output_dim,
                num_mlp = 4,
                without_norm = True       
            ) for _ in range(self.num_decoder_layers+1)])
        
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
        if self.loss_mode == 'best':
            return self.get_loss_best(tb_pre_tag)
        else:
            return self.get_loss_gmm(tb_pre_tag)

    def get_loss_best(self, tb_pre_tag='', filter = True):
        tb_dict = {}
        
        center_gt = self.forward_ret_dict['center_gt'][...,None,:3].cuda()
        # normalize the gt
        center_gt = (center_gt - self.output_mean) / self.output_std
        
        total_loss = 0 
        for i, (pred_states, pred_scores) in enumerate(self.forward_ret_dict['pred_list']):               
            # Get mode for all
            # print(pred_states.shape)
            mode_all = self.build_mode_distribution(pred_states) # [batch size]
            nll_loss_all = -mode_all.log_prob(center_gt)            
            nll_loss_best, best_idx = nll_loss_all.min(dim=-1)
            
            # Filter out the noise prediction
            if filter:
                nll_loss_valid = nll_loss_best < 20
                nll_loss_best = nll_loss_best[nll_loss_valid]
                best_idx = best_idx[nll_loss_valid]
                pred_scores = pred_scores[nll_loss_valid]
                tb_dict[f'{tb_pre_tag}layer{i}_num_invalid'] = torch.sum(~nll_loss_valid).float().item()

            cls_loss = F.cross_entropy(input  = pred_scores, target= best_idx, reduction='none')

            layer_loss = (nll_loss_best + cls_loss).mean()
            tb_dict[f'{tb_pre_tag}layer{i}_loss_nll'] = nll_loss_best.mean().item()
            tb_dict[f'{tb_pre_tag}layer{i}_loss_cls'] = cls_loss.mean().item()
            tb_dict[f'{tb_pre_tag}layer{i}_loss'] = layer_loss.item()
            
            tb_dict[f'{tb_pre_tag}layer{i}_pred_mean'] = pred_states.mean().item()
            tb_dict[f'{tb_pre_tag}layer{i}_pred_std'] = pred_states.std(dim=1).mean().item()
            tb_dict[f'{tb_pre_tag}layer{i}_score_mean'] = pred_scores.mean().item()
            tb_dict[f'{tb_pre_tag}layer{i}_score_std'] = pred_scores.std(dim=1).mean().item()
            
            total_loss += layer_loss
        
        # Average over layers    
        total_loss /= len(self.forward_ret_dict['pred_list'])
        
        tb_dict[f'{tb_pre_tag}loss_total'] = total_loss.item()
            
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

        total_loss = nll_loss.mean()
        tb_dict[f'{tb_pre_tag}loss_nll'] = nll_loss.mean().item()
        tb_dict[f'{tb_pre_tag}loss_total'] = total_loss.item()
        
        # record mean and std
        tb_dict[f'{tb_pre_tag}pred_mean'] = pred_ctrls.mean().item()
        tb_dict[f'{tb_pre_tag}pred_std'] = pred_ctrls.std().item()
        tb_dict[f'{tb_pre_tag}score_mean'] = pred_scores.mean().item()
        tb_dict[f'{tb_pre_tag}score_std'] = pred_scores.std().item()
        return total_loss, tb_dict
    
    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        
        # Aggregate features over the history 
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']
        track_index_to_predict = batch_dict['track_index_to_predict']

        num_center_objects, num_objects, _ = obj_feature.shape

        num_polylines = map_feature.shape[1]
        
        # Remove Ego agent from the object feature
        # obj_mask[torch.arange(num_center_objects), track_index_to_predict] = False
        
        # input projection 
        # project each feature to a higher dimension
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        center_objects_feature = center_objects_feature[None,...].repeat(self.num_motion_modes, 1, 1) # (1, num_center_objects, C)
        
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid
        obj_feature = obj_feature.permute(1, 0, 2).contiguous() # (num_objects, num_center_objects, C)

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid
        map_feature = map_feature.permute(1, 0, 2).contiguous() # (num_polylines, num_center_objects, C)
        
        # Get positional embedding of the query
        obj_pos_embed = position_encoding_utils.gen_sineembed_for_position(
            obj_pos.permute(1, 0, 2)[:, :, 0:2], hidden_dim=self.d_model
        ).contiguous() # (num_objects, num_center_objects, C)
        
        map_pos_embed = position_encoding_utils.gen_sineembed_for_position(
            map_pos.permute(1, 0, 2)[:, :, 0:2], hidden_dim=self.d_model
        ).contiguous() # (num_polylines, num_center_objects, C)
        
        center_pos_embed = obj_pos_embed[track_index_to_predict, torch.arange(num_center_objects), :] # (num_center_objects, C)
        center_pos_embed = center_pos_embed.unsqueeze(0).repeat(self.num_motion_modes, 1, 1) # (num_motion_modes, num_center_objects, C)
        
        # Process the query
        query_embed = self.query  # (Q, C)
        # query_embed.register_hook(print)
        query_embed = query_embed.unsqueeze(1).repeat(1, num_center_objects, 1)  # (num_motion_modes, num_center_objects, C)
        
        query_embed = self.pre_query_fusion_layer(torch.cat([center_objects_feature, query_embed], dim=-1))
        
        # Initialize prediction with out attention
        prediction = self.prediction_layers[0](query_embed)
        pred_scores = prediction[..., -1].permute(1, 0).contiguous()
        pred_states = prediction[..., :-1].permute(1, 0, 2).contiguous() # (num_center_objects, num_motion_modes, 9)
        
        if self.use_bicycle_model:
            pred_ctrls = pred_states[..., :2] # (num_center_objects, num_motion_modes, 2)
            pred_ctrls_sigma = pred_states[..., 2:] # (num_center_objects, num_motion_modes, 6)
            pred_states = self.bicycle_forward(input_dict, pred_ctrls)
            pred_states = (pred_states - self.output_mean) / self.output_std # normalize the output
            pred_states = torch.cat([pred_states, pred_ctrls_sigma], dim=-1) # (num_center_objects, num_motion_modes, 9)
        
        pred_list = [(pred_states, pred_scores)]
        
        
        for i in range(self.num_decoder_layers):
            obj_atten = self.obj_atten_layers[i]
            map_atten = self.map_atten_layers[i]
            query_fuison = self.query_fusion_layers[i]
            pred_layer = self.prediction_layers[i+1]
            
            obj_query_embed = obj_atten(
                tgt = query_embed,
                memory = obj_feature,
                tgt_mask = None,
                memory_mask = None,
                tgt_pos = center_pos_embed,
                memory_pos = obj_pos_embed,
                memory_key_padding_mask = ~obj_mask,
            )
            
            map_query_embed = map_atten(
                tgt = query_embed,
                memory = map_feature,
                tgt_mask = None,
                memory_mask = None,
                tgt_pos = center_pos_embed,
                memory_pos = map_pos_embed,
                memory_key_padding_mask = ~map_mask,
            )
            
            # print("query_embed", query_embed.std())
            temp = query_fuison(
                torch.cat([query_embed,
                        obj_query_embed,
                        map_query_embed
                    ], dim=-1)) 
            
            # print("temp", temp.std())
            query_embed = temp #+ query_embed
            
            prediction = pred_layer(query_embed)
            
            pred_states = prediction[..., :-1].permute(1, 0, 2).contiguous() # (num_center_objects, num_motion_modes, 9)
            pred_scores = prediction[..., -1].permute(1, 0).contiguous()
            
            if self.use_bicycle_model:
                pred_ctrls = pred_states[..., :2] # (num_center_objects, num_motion_modes, 2)
                pred_ctrls_sigma = pred_states[..., 2:] # (num_center_objects, num_motion_modes, 6)
                pred_states = self.bicycle_forward(input_dict, pred_ctrls)
                pred_states = (pred_states - self.output_mean) / self.output_std # normalize the output
                pred_states = torch.cat([pred_states, pred_ctrls_sigma], dim=-1) # (num_center_objects, num_motion_modes, 9)
            
            pred_list.append((pred_states, pred_scores))
            
        if 'center_gt' in input_dict:
            self.forward_ret_dict['pred_list'] = pred_list
            self.forward_ret_dict['center_gt'] = input_dict['center_gt']
            # Otherwise, it is in the inference mode
            
        batch_dict['pred_list'] = pred_list
        return batch_dict
    
    def bicycle_forward(self, input_dict, pred_ctrls, dt = 0.1, step = 5):
        
        num_center, num_modes, _ = pred_ctrls.shape
        
        full_traj = input_dict['obj_trajs'].cuda()
        center_idx = input_dict['track_index_to_predict']
        center_traj = full_traj[torch.arange(num_center), center_idx, -1]
        
        
        # initialize the state
        Length = 5 #center_traj[..., 3, None]
        v = center_traj[..., -4:-2].norm(dim=-1, keepdim=True)
        x = torch.zeros_like(v, device=v.device)
        y = torch.zeros_like(v, device=v.device)
        theta = torch.zeros_like(v, device=v.device)
        
        a = pred_ctrls[..., 0] * 10 # max acceleration is 10 m/s^2
        delta = pred_ctrls[..., 1] * 0.35 # max steering angle is 0.35 rad        
        beta = torch.arctan(0.5*torch.tan(delta))        
        ddt = dt / step
        for _ in range(step):
            # print(v.shape, delta.shape, Length.shape)
            theta = theta + v * torch.sin(delta) * ddt / (Length/2)
            x = x + v * torch.cos(theta+beta) * ddt
            y = y + v * torch.sin(theta+beta) * ddt
            v = v + a * ddt
            
        return torch.stack([x, y, theta], dim=-1)
        
        
        
