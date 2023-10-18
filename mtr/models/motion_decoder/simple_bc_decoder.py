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
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.use_place_holder = self.model_cfg.get('USE_PLACE_HOLDER', False)
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS

        # define the cross-attn layers
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=self.d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=False
        )

        map_d_model = self.model_cfg.get('MAP_D_MODEL', self.d_model)
        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=map_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=False
        )
        if map_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, map_d_model)
            self.map_query_content_mlps = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            self.map_query_embed_mlps = nn.Linear(self.d_model, map_d_model)
        else:
            self.map_query_content_mlps = [None for _ in range(self.num_decoder_layers)]
            self.map_query_embed_mlps = None
            
        # Create a place holder for the motion query
        self.query = nn.Parameter(torch.rand(self.num_motion_modes, self.d_model), requires_grad=True)

        self.query_mlps = common_layers.build_mlps(
            c_in=self.d_model, mlp_channels=[self.d_model, self.d_model], ret_before_act=True
        )

        # define the motion head
        temp_layer = common_layers.build_mlps(c_in=self.d_model * 2 + map_d_model, mlp_channels=[self.d_model, self.d_model], ret_before_act=True)
        self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        self.motion_reg_heads, self.motion_cls_heads = self.build_motion_head(
            in_channels=self.d_model, hidden_size=self.d_model, num_decoder_layers=self.num_decoder_layers
        )
        
        self.register_buffer('output_mean', torch.tensor([0.65429819, 0.00128331, 0.0006909]))
        self.register_buffer('output_std', torch.tensor([0.54449782, 0.02267612, 0.05700458]))
        
        self.forward_ret_dict = {}

    def build_transformer_decoder(self, in_channels, d_model, nhead, dropout=0.1, num_decoder_layers=1, use_local_attn=False):
        in_proj_layer = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = transformer_decoder_layer.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=False,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn
        )
        decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        return in_proj_layer, decoder_layers

    def build_motion_head(self, in_channels, hidden_size, num_decoder_layers):
        motion_reg_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size,  9], ret_before_act=True
        )
        motion_cls_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        )

        motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
        return motion_reg_heads, motion_cls_heads

    def apply_cross_attention(self, kv_feature, kv_mask, kv_pos, query_content, query_embed, attention_layer,
                              dynamic_query_center=None, layer_idx=0, use_local_attn=False, query_index_pair=None,
                              query_content_pre_mlp=None, query_embed_pre_mlp=None):
        """
        Args:
            kv_feature (B, N, C): Key and Value feature
            kv_mask (B, N): Key and Value mask
            kv_pos (B, N, 3): Ego centric x-y position of Key and Value. This is used for positional encoding.
            query_content (M, B, C): # Query feature
            query_embed (M, B, C): # Query positional embedding 
            attention_layer (layer): nn.Module Attention layer to be used.
            dynamic_query_center (M, B, 2): . Ego centric x-y position of query center. This is used for positional encoding.
            layer_idx: int. Defaults to 0.
            use_local_attn (bool): Whether to use local attention. Defaults to False.
            query_index_pair (B, M, K)

        Returns:
            attended_features: (B, M, C)
            attn_weights:
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        # These two positional embeddings are used for the cross attention, should be aligned 
        num_q, batch_size, d_model = query_content.shape
        searching_query = position_encoding_utils.gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)
        kv_pos = kv_pos.permute(1, 0, 2)[:, :, 0:2]
        kv_pos_embed = position_encoding_utils.gen_sineembed_for_position(kv_pos, hidden_dim=d_model)

        if not use_local_attn:
            query_feature = attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                query_sine_embed=searching_query,
                memory=kv_feature.permute(1, 0, 2),
                memory_key_padding_mask=~kv_mask,
                pos=kv_pos_embed,
                is_first=(layer_idx == 0)
            )  # (M, B, C)
        else:
            batch_size, num_kv, _ = kv_feature.shape

            kv_feature_stack = kv_feature.flatten(start_dim=0, end_dim=1)
            kv_pos_embed_stack = kv_pos_embed.permute(1, 0, 2).contiguous().flatten(start_dim=0, end_dim=1)
            kv_mask_stack = kv_mask.view(-1)

            key_batch_cnt = num_kv * torch.ones(batch_size).int().to(kv_feature.device)
            query_index_pair = query_index_pair.view(batch_size * num_q, -1)
            index_pair_batch = torch.arange(batch_size).type_as(key_batch_cnt)[:, None].repeat(1, num_q).view(-1)  # (batch_size * num_q)
            assert len(query_index_pair) == len(index_pair_batch)

            query_feature = attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                query_sine_embed=searching_query,
                memory=kv_feature_stack,
                memory_valid_mask=kv_mask_stack,
                pos=kv_pos_embed_stack,
                is_first=(layer_idx == 0),
                key_batch_cnt=key_batch_cnt,
                index_pair=query_index_pair,
                index_pair_batch=index_pair_batch
            )
            query_feature = query_feature.view(batch_size, num_q, d_model).permute(1, 0, 2)  # (M, B, C)

        return query_feature

    def apply_transformer_decoder(self, center_objects_feature, query_embed, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos):
        # Encoded and raw position of intention points
        # ! TODO, use our own intention points
        
        query_content = torch.zeros_like(query_embed) #[Num Query, Batch, C]
        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]
        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)  # (num_query, num_center_objects, C)
        dynamic_query_center = torch.zeros((num_query, num_center_objects, 2), device=query_content.device)  # (num_query, num_center_objects, 3)

        pred_list = []
        # query_embed.register_hook(print)
        for layer_idx in range(self.num_decoder_layers):
            # print(dynamic_query_center[:, 1, :])
            # ! Why initial query_content is all zeros?
            # query object feature
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature, kv_mask=obj_mask, kv_pos=obj_pos,
                query_content=query_content, query_embed=query_embed,
                attention_layer=self.obj_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx
            ) 
            map_query_feature = self.apply_cross_attention(
                kv_feature=map_feature, kv_mask=map_mask, kv_pos=map_pos,
                query_content=query_content, query_embed=query_embed,
                attention_layer=self.map_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx,
                query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_pre_mlp=self.map_query_embed_mlps
            ) 
            

            query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1) 

            # motion prediction
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            pred_ctrls = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, 9)
            
            pred_list.append([pred_scores, pred_ctrls])
            
            # update
            dynamic_query_center = pred_ctrls[:, :, :2].contiguous().permute(1, 0, 2)  # (num_query, num_center_objects, 2)

        if self.use_place_holder:
            raise NotImplementedError

        assert len(pred_list) == self.num_decoder_layers
        return pred_list
    

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
        # print(covariance)
        mode = MultivariateNormal(mean, covariance_matrix=covariance)
      
        return mode
    
    
    def build_gmm_distribution(self, pred_ctrl, pred_ctrl_score, log_std_range=(-5.0, 2.0), rho_limit=0.4):
        mode = self.build_mode_distribution(pred_ctrl, log_std_range, rho_limit)
        mix = Categorical(logits=pred_ctrl_score)
        gmm = MixtureSameFamily(mix, mode)

    def get_loss(self, tb_pre_tag=''):
        center_gt = self.forward_ret_dict['center_gt'][...,:3].cuda()
        # normalize the gt
        center_gt = (center_gt - self.output_mean) / self.output_std
        center_gt = center_gt.unsqueeze(1) #[b, 1, 3]

        pred_list = self.forward_ret_dict['pred_list']
        
        tb_dict = {}
        total_loss = 0
        for layer_idx in range(self.num_decoder_layers):
            pred_scores, pred_ctrls = pred_list[layer_idx]
            best_idx = (pred_ctrls[...,:3] - center_gt).norm(dim=-1).argmin(dim=-1)
            pred_ctrls_best = pred_ctrls[torch.arange(pred_ctrls.shape[0]), best_idx]
            
            mode = self.build_mode_distribution(pred_ctrls_best) # [batch size]
            
            nll_loss = -mode.log_prob(center_gt)
            
            # cross entropy loss
            cls_loss = F.cross_entropy(input  = pred_scores, target= best_idx, reduction='none')
            
            layer_loss = nll_loss + cls_loss.sum(dim=-1)
            layer_loss = layer_loss.mean()
            total_loss += layer_loss
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}'] = layer_loss.item()
            
        total_loss = total_loss / self.num_decoder_layers
        return total_loss, tb_dict
   
    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        
        # Aggregate features over the history 
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape

        num_polylines = map_feature.shape[1]
        
        # input projection 
        # project each feature to a higher dimension
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid
        
        # Process the query
        # self.query.register_hook(print)
        query_embed = self.query_mlps(self.query)  # (Q, C)
        # query_embed.register_hook(print)

        query_embed = query_embed.unsqueeze(1).repeat(1, num_center_objects, 1)  # (Q, N, C)
        # decoder layers
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            query_embed=query_embed,
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos,
        )
        
        # Generate 9D Control
        # [dx, dy, dtheta, sigma_x, signa_y, sigma_theta, rho_xy, rho_xtheta, rho_ytheta]
        if 'center_gt' in input_dict:
            self.forward_ret_dict['pred_list'] = pred_list
            self.forward_ret_dict['center_gt'] = input_dict['center_gt']
            # Otherwise, it is in the inference mode
            
        batch_dict['pred_list'] = pred_list

        return batch_dict
