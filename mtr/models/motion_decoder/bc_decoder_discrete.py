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
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical

import matplotlib.pyplot as plt
                   
class BCDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.d_model = self.model_cfg.D_MODEL
        self.n_head = self.model_cfg.NUM_ATTN_HEAD
        self.dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1)
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS
        self.num_accel_grid = self.model_cfg.NUM_ACCEL_GRID
        self.num_steer_grid = self.model_cfg.NUM_STEER_GRID
        self.num_motion_modes = self.num_accel_grid * self.num_steer_grid
        self.pred_all_layers = self.model_cfg.get('PRED_ALL_LAYERS', True)
        
        self.accel_grid = nn.Parameter(
            torch.linspace(-10, 10, self.num_accel_grid),
            requires_grad= False
        )
        self.steer_grid = nn.Parameter(
            torch.linspace(-0.3, 0.3, self.num_steer_grid),
            requires_grad= False
        )
            
        self.accel_embed = nn.Embedding(self.num_accel_grid, self.d_model)
        self.steer_embed = nn.Embedding(self.num_steer_grid, self.d_model)
        
        # Build the query
        self.accel_query = nn.Parameter(
            torch.tensor([i for i in range(self.num_accel_grid) for _ in range(self.num_steer_grid)], dtype = torch.int32),
            requires_grad= False
        )
        self.steer_query = nn.Parameter(
            torch.tensor([i for _ in range(self.num_accel_grid) for i in range(self.num_steer_grid)], dtype = torch.int32),
            requires_grad= False
        )
        self.in_proj_query = nn.Sequential(
            nn.Linear(2*self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
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
        output_dim = 1
            
        self.prediction_layers = nn.ModuleList([ResidualMLP(
                c_in = self.d_model,
                c_out = output_dim,
                num_mlp = 4,
                without_norm = True       
            ) for _ in range(self.num_decoder_layers+1)])
        
        self.register_buffer('output_mean', torch.tensor([0.0, 0.0]))
        self.register_buffer('output_std', torch.tensor([3.5, 0.17]))
        
    def get_loss(self, decoder_dict, tb_pre_tag='', debug = False):
        tb_dict = {}
        
        center_gt = decoder_dict['gt_action'] # [batch size, 2]
        
        # Find idx
        accel_idx = torch.searchsorted(self.accel_grid, center_gt[:, 0])
        steer_idx = torch.searchsorted(self.steer_grid, center_gt[:, 1])
        best_idx = accel_idx * self.num_steer_grid + steer_idx
        
        total_loss = 0
        if debug:
            _, axs = plt.subplots(1, len(decoder_dict['pred_list']), figsize=(10, 10))
        for i, pred_logits in enumerate(decoder_dict['pred_list']):    
            layer_loss = F.cross_entropy(input=pred_logits, target=best_idx, reduction='mean')
            tb_dict[f'{tb_pre_tag}layer{i}_loss'] = layer_loss.item()            
            total_loss += layer_loss
            if debug:
                prob = F.softmax(pred_logits, dim=-1)
                im = axs[i].contourf(
                    self.accel_grid.detach().cpu().numpy(),
                    self.steer_grid.detach().cpu().numpy(),
                    prob.reshape((self.num_accel_grid, self.num_steer_grid)).detach().cpu().numpy(),
                    # aspect='square'
                )
                axs[i].set_aspect(10/0.3)
                plt.colorbar(im, ax=axs[i], shrink=0.2, aspect=20)
                axs[i].set_title(f'layer {i}')
        
        
        # Average over layers    
        total_loss /= len(decoder_dict['pred_list'])
        tb_dict[f'{tb_pre_tag}loss_total'] = total_loss.item()
            
        return total_loss, tb_dict

    
    def forward(self, batch_dict):
        # Aggregate features over the history 
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'].cuda(), batch_dict['obj_mask'].cuda(), batch_dict['obj_pos'].cuda()
        map_feature, map_mask, map_pos = batch_dict['map_feature'].cuda(), batch_dict['map_mask'].cuda(), batch_dict['map_pos'].cuda()
        track_index_to_predict = batch_dict['track_index_to_predict'].cuda()
        
        num_center_objects, num_objects, _ = obj_feature.shape
        
        center_objects_feature = obj_feature[torch.arange(num_center_objects), track_index_to_predict]

        num_polylines = map_feature.shape[1]
        
        # Remove Ego agent from the object feature
        # obj_mask[torch.arange(num_center_objects), track_index_to_predict] = False
        
        # Generate query embedding
        accel_embed = self.accel_embed(self.accel_query.cuda())
        steer_embed = self.steer_embed(self.steer_query.cuda())
        query_embed = torch.cat([accel_embed, steer_embed], dim=-1)
        query_embed = self.in_proj_query(query_embed)
        
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
        # query_embed.register_hook(print)
        query_embed = query_embed.unsqueeze(1).repeat(1, num_center_objects, 1)  # (num_motion_modes, num_center_objects, C)
        
        query_embed = self.pre_query_fusion_layer(torch.cat([center_objects_feature, query_embed], dim=-1))
        
        pred_list = []
        if self.pred_all_layers:
            # Initialize prediction with out attention
            prediction = self.prediction_layers[0](query_embed).permute(1, 0, 2).squeeze(-1).contiguous() # (num_center_objects, num_query, 1)
            pred_list.append(prediction)
        
        
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
            
            if self.pred_all_layers or i == (self.num_decoder_layers - 1):
                prediction = pred_layer(query_embed).permute(1, 0, 2).squeeze(-1).contiguous() # (num_center_objects, num_query, 1)        
                pred_list.append(prediction)
                
        batch_dict['pred_list'] = pred_list
        return batch_dict
    
    def load_model(
        self,
        state_dict: dict
    ):
        
        model_keys = self.state_dict().keys()
        
        state_dict_filtered = {}
        # search for the weights in the state_dict_to_load and save to a new dict
        for key in model_keys:
            for state_key in state_dict.keys():
                if state_key.endswith(key):
                    state_dict_filtered[key] = state_dict[state_key]
                    break
                
        # load the filtered state_dict
        self.load_state_dict(state_dict_filtered, strict=True)