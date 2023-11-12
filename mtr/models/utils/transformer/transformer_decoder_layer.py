# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Modified by Shaoshuai Shi 
# All Rights Reserved


"""
Modified from https://github.com/IDEA-opensource/DAB-DETR/blob/main/models/DAB_DETR/transformer.py
"""

from typing import Optional, List
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from .transformer_encoder_layer import _get_activation_fn
# from .multi_head_attention_local import MultiheadAttentionLocal
from .multi_head_attention import MultiheadAttention


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
            self.self_atten = MultiheadAttention(
                embed_dim=d_model, num_heads=n_head, dropout=dropout, vdim=d_model, without_weight=True
            )
            self.self_atten_dropout = nn.Dropout(dropout)
        
        self.cross_atten = MultiheadAttention(
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