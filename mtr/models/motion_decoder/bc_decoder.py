# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

from typing import Optional, List

import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtr.models.utils.transformer import transformer_decoder_layer, multi_head_attention
from mtr.models.utils.transformer import position_encoding_utils
from mtr.models.utils import common_layers
from mtr.utils import common_utils, loss_utils, motion_utils
# from mtr.config import cfg
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, with_self_atten = True) -> None:
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "gelu":
            self.activation = nn.GELU
        else:
            self.activation = nn.LeakyReLU
            
        self.n_head = n_head
        self.normalize_before = normalize_before
        self.with_self_atten = with_self_atten
        
        if self.normalize_before:
            template = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
            )
            # In the pre-norm case, follows the order of norm -> FFN -> dropout -> add
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim_feedforward),
                self.activation(),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            
        else:
            template = nn.Linear(d_model, d_model)
            # In the original paper, follows the order of FFN -> dropout -> add -> norm
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                self.activation(),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            if self.with_self_atten:
                self.sa_post_norm = nn.LayerNorm(d_model)
            self.ca_post_norm = nn.LayerNorm(d_model)
            self.ffn_post_norm = nn.LayerNorm(d_model)
            
        if self.with_self_atten:    
            self.sa_q_proj = copy.deepcopy(template)
            self.sa_q_pos_proj = copy.deepcopy(template)
            self.sa_k_proj = copy.deepcopy(template)
            self.sa_k_pos_proj = copy.deepcopy(template)
            self.sa_v_proj = copy.deepcopy(template)
        
        self.ca_q_proj = copy.deepcopy(template)
        self.ca_q_pos_proj = copy.deepcopy(template)
        self.ca_k_proj = copy.deepcopy(template)
        self.ca_k_pos_proj = copy.deepcopy(template)
        self.ca_v_proj = copy.deepcopy(template)
            
        if self.with_self_atten:
            self.self_atten = multi_head_attention.MultiheadAttention(
                embed_dim=d_model, num_heads=n_head, dropout=dropout, vdim=d_model, without_weight=True
            )
            self.self_atten_dropout = nn.Dropout(dropout)
        
        self.cross_atten = multi_head_attention.MultiheadAttention(
            embed_dim=2*d_model, num_heads=n_head, dropout=dropout, vdim=d_model, without_weight=True
        )
        self.cross_atten_dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):    
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def multi_head_concat(self, embed, pos_embed):
        N, B, D = embed.shape
        embed = embed.view(N, B, self.n_head, D // self.n_head)
        pos_embed = pos_embed.view(N, B, self.n_head, D // self.n_head)
        embed_concat = torch.cat([embed, pos_embed], dim=-1)
        embed_concat = embed_concat.view(N, B, -1)
        return embed_concat
                
    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_pos: Optional[torch.Tensor] = None,
                memory_pos: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor]=None,
    ):
        '''
        Args:
            tgt (num_query, B, C): This is the query
            memory (M, B, C): Key and value
            tgt_mask (num_query, B): Mask for the query
            memory_mask (M, B): Mask for the key and value
            query_pos (num_query, B, C): Positional embedding of the query 
            memory_pos (M, B, C): Positional embedding of the key and value
            memory_key_padding_mask (M, B): Mask for the key and value
        '''
        if self.with_self_atten:
            # q_sub = self.multi_head_concat(self.sa_q_proj(tgt), self.sa_q_pos_proj(tgt_pos))
            # k_sub = self.multi_head_concat(self.sa_k_proj(tgt), self.sa_k_pos_proj(tgt_pos))
            q_sub = self.sa_q_proj(tgt)
            k_sub = self.sa_k_proj(tgt)
            v_sub = self.sa_v_proj(tgt)

            tgt_sub = self.self_atten(q_sub, k_sub, value = v_sub,
                                      attn_mask=tgt_mask, key_padding_mask=None)[0]
            tgt = tgt + self.self_atten_dropout(tgt_sub)
            if not self.normalize_before:
                tgt = self.sa_post_norm(tgt)
        
        # Cross Attention
        q_sub = self.multi_head_concat(self.ca_q_proj(tgt), self.ca_q_pos_proj(tgt_pos))
        # q_sub = self.ca_q_proj(tgt)
        k_sub = self.multi_head_concat(self.ca_k_proj(memory), self.ca_k_pos_proj(memory_pos))
        # k_sub = self.ca_k_proj(memory) + self.ca_k_pos_proj(memory_pos)
        v_sub = self.ca_v_proj(memory)
        
        tgt_sub = self.cross_atten(q_sub, k_sub, value = v_sub,
                                   attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.cross_atten_dropout(tgt_sub)
        if not self.normalize_before:
            tgt = self.ca_post_norm(tgt)
        
        # FFN
        tgt = tgt+self.ffn(tgt_sub)
        if not self.normalize_before:
            tgt = self.ffn_post_norm(tgt)
            
        return tgt

class ResidualMLP(nn.Module):
    def __init__(self, c_in, c_out, num_mlp=4, without_norm = True) -> None:
        super().__init__()
        assert num_mlp >= 1
        
        self.mlp_list = nn.ModuleList(
            [
                common_layers.build_mlps(
                    c_in=c_in,
                    mlp_channels=[c_in, c_in],
                    without_norm=without_norm,
                    ret_before_act=True
                )
                for _ in range(num_mlp)
            ]
        )
        self.out_mlp = common_layers.build_mlps(
            c_in=c_in,
            mlp_channels=[c_in, c_out],
            without_norm=without_norm,
            ret_before_act=True
        )
        
    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        for mlp in self.mlp_list:
            x = x + mlp(x)
        return self.out_mlp(x).view(*original_shape[:-1], -1)
                         
class BCDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.use_place_holder = self.model_cfg.get('USE_PLACE_HOLDER', False)
        self.d_model = self.model_cfg.D_MODEL
        self.n_head = self.model_cfg.NUM_ATTN_HEAD
        self.dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1)
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS
        self.loss_mode = self.model_cfg.get('LOSS_MODE', 'best')
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        
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
        self.prediction_layers = nn.ModuleList([ResidualMLP(
                c_in = self.d_model,
                c_out = 10,
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
        for i, (pred_ctrls, pred_scores) in enumerate(self.forward_ret_dict['pred_list']):
            # Get mode for all
            mode_all = self.build_mode_distribution(pred_ctrls) # [batch size]
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
            
            tb_dict[f'{tb_pre_tag}layer{i}_pred_mean'] = pred_ctrls.mean().item()
            tb_dict[f'{tb_pre_tag}layer{i}_pred_std'] = pred_ctrls.std(dim=1).mean().item()
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
        pred_ctrls = prediction[..., :-1].permute(1, 0, 2).contiguous() # (num_center_objects, num_motion_modes, 9)
        pred_scores = prediction[..., -1].permute(1, 0).contiguous()
        pred_list = [(pred_ctrls, pred_scores)]
        
        
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
            
            pred_ctrls = prediction[..., :-1].permute(1, 0, 2).contiguous() # (num_center_objects, num_motion_modes, 9)
            pred_scores = prediction[..., -1].permute(1, 0).contiguous()
            
            pred_list.append((pred_ctrls, pred_scores))
            
        if 'center_gt' in input_dict:
            self.forward_ret_dict['pred_list'] = pred_list
            self.forward_ret_dict['center_gt'] = input_dict['center_gt']
            # Otherwise, it is in the inference mode
            
        batch_dict['pred_list'] = pred_list
        return batch_dict
