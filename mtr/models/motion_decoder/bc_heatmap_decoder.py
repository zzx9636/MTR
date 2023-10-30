from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

from mtr.models.utils.transformer.position_encoding_utils import gen_sineembed_for_position
from mtr.models.motion_decoder.bc_decoder import TransformerDecoder, ResidualMLP
from mtr.utils.loss_utils import (
    pixelwise_focal_loss, pixelwise_home_loss, pixelwise_target_entropy_loss, gen_smooth_heatmap_target
)
from mtr.utils.motion_utils import bicycle_RK4


class BCHeatmapDecoder(nn.Module):

    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.use_place_holder = self.model_cfg.get('USE_PLACE_HOLDER', False)
        self.d_model = self.model_cfg.D_MODEL
        self.n_head = self.model_cfg.NUM_ATTN_HEAD
        self.dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1)
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS
        self.loss_mode = self.model_cfg.get('LOSS_MODE', 'best')
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)

        # Build the query (basically just indices for control)
        accels = torch.arange(
            self.model_cfg.ACCEL_MIN, self.model_cfg.ACCEL_MAX + self.model_cfg.ACCEL_SPACE, self.model_cfg.ACCEL_SPACE
        )
        steerings = torch.arange(
            self.model_cfg.STEER_MIN, self.model_cfg.STEER_MAX + self.model_cfg.STEER_SPACE, self.model_cfg.STEER_SPACE
        )
        grid_x, grid_y = torch.meshgrid(accels, steerings, indexing='ij')
        ctrl_grids = torch.stack([grid_x, grid_y], dim=-1)
        self.ctrl_grids = ctrl_grids.reshape(-1, 2)
        self.num_accels = len(accels)
        self.num_steers = len(steerings)
        self.num_motion_modes = self.ctrl_grids.shape[0]

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1., 1., self.num_accels), torch.linspace(-1., 1., self.num_steers), indexing='ij'
        )
        idx_grids = torch.stack([grid_x, grid_y], dim=-1)
        self.query = idx_grids.reshape(-1, 2)
        # self.register_buffer('query', query)
        # self.register_buffer('ctrl_grids', ctrl_grids)

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

        self.in_proj_query = nn.Sequential(  # from controls to embeddings
            nn.Linear(2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Query Fusion
        self.pre_query_fusion_layer = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.query_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model * 3, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            ) for _ in range(self.num_decoder_layers)
        ])

        # Attention layers
        self.obj_atten_layers = nn.ModuleList([
            TransformerDecoder(
                d_model=self.d_model, n_head=self.n_head, dim_feedforward=self.d_model * 4, dropout=self.dropout,
                with_self_atten=False, normalize_before=True
            ) for _ in range(self.num_decoder_layers)
        ])

        self.map_atten_layers = nn.ModuleList([
            TransformerDecoder(
                d_model=self.d_model, n_head=self.n_head, dim_feedforward=self.d_model * 4, dropout=self.dropout,
                with_self_atten=False, normalize_before=True
            ) for _ in range(self.num_decoder_layers)
        ])

        self.ctrl_atten_layers = nn.ModuleList([
            TransformerDecoder(
                d_model=self.d_model, n_head=self.n_head, dim_feedforward=self.d_model * 4, dropout=self.dropout,
                with_self_atten=False, normalize_before=True
            ) for _ in range(self.num_decoder_layers)
        ])

        # Prediction Head
        self.prediction_layers = nn.ModuleList([
            nn.Sequential(
                ResidualMLP(c_in=self.d_model, c_out=1, num_mlp=2, without_norm=True),
                nn.Softmax(dim=0),
            ) for _ in range(self.num_decoder_layers + 1)
        ])

        # Loss
        self.loss_alpha = self.model_cfg.LOSS_ALPHA  # focal loss: focuses more on the hard examples.
        self.loss_beta = self.model_cfg.LOSS_BETA  # reduces penalties for pixels close to the ground truth.
        self.loss_gaussian_std = self.model_cfg.LOSS_GAUSSIAN_STD  # smoothes heatmap.

        self.forward_ret_dict = {}
        self.debug = False

    def get_indices_gt(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_cur = self.forward_ret_dict['state_cur'][:, None].cuda()  # (num_center_objects, 1, 4)
        controls = self.ctrl_grids[None].cuda()  # (1, num_ctrls, 2)
        state_nxt = bicycle_RK4(state_cur, controls, dt=0.1, car_length=self.forward_ret_dict['car_length'].cuda())
        pos = state_nxt[..., :2]

        center_gt = self.forward_ret_dict['center_gt'][:, None, :2].cuda()  # (num_center_objects, 1, 2)
        distance = torch.linalg.norm(pos - center_gt, dim=-1)  # (num_center_objects, num_motion_modes)
        best_dist, best_idx = distance.min(dim=-1)
        best_idx_tuple = torch.zeros(center_gt.shape[0], 2, dtype=int)
        best_idx_tuple[:, 0] = best_idx // self.num_steers
        best_idx_tuple[:, 1] = best_idx % self.num_steers
        return best_dist, best_idx_tuple, state_nxt

    def get_loss(self, tb_pre_tag='') -> Tuple[torch.Tensor, dict]:
        best_dist, best_idx_tuple, state_nxt = self.get_indices_gt()
        mask = best_dist < 1.0  # filters out if there is no control that leads to close prediction.
        target = gen_smooth_heatmap_target(
            self.num_accels, self.num_steers, best_idx_tuple, gaussian_std=self.loss_gaussian_std
        )

        total_loss = 0
        if self.debug:
            tb_dict = {
                'state_nxt': state_nxt.cpu().numpy(),
                'heatmap': target.cpu().numpy(),
                'best_idx_tuple': best_idx_tuple.cpu().numpy()
            }
        else:
            tb_dict = {}
        for i, pred_probs in enumerate(self.forward_ret_dict['pred_list']):
            # _, loss_all = pixelwise_focal_loss(
            #     pred_probs, best_idx_tuple, alpha=self.loss_alpha, beta=self.loss_beta,
            #     gaussian_std=self.loss_gaussian_std, target=target
            # )
            # _, loss_all = pixelwise_home_loss(
            #     pred_probs, best_idx_tuple, beta=self.loss_beta, gaussian_std=self.loss_gaussian_std, target=target
            # )
            _, loss_all = pixelwise_target_entropy_loss(pred_probs, best_idx_tuple, target=target)
            loss_all *= mask[:, None, None].to(loss_all)
            loss_mean = loss_all.mean()

            tb_dict[f'{tb_pre_tag}layer{i}_loss'] = loss_mean.item()
            if self.debug:
                tb_dict[f'{tb_pre_tag}layer{i}_pred_probs'] = pred_probs.detach().cpu().numpy()
                tb_dict[f'{tb_pre_tag}layer{i}_loss_all'] = loss_all.detach().cpu().numpy()
            total_loss += loss_mean

        # Average over layers
        total_loss /= len(self.forward_ret_dict['pred_list'])
        tb_dict[f'{tb_pre_tag}loss_total'] = total_loss.item()
        return total_loss, tb_dict

    def sample(self, batch_dict) -> np.ndarray:
        """

        Args:
            batch_dict (_type_): _description_

        Returns:
            np.ndarray: next pose of the center objects in the body frame (x, y, heading).
        """
        self.forward(batch_dict)
        pred_probs: torch.Tensor = batch_dict['pred_list'][-1].detach().cpu()
        state_cur = batch_dict['state_cur'][:, None]  # (num_center_objects, 1, 4)
        car_length = batch_dict['car_length']

        num_center_objects, width, height = pred_probs.shape

        pred_probs_flattened = pred_probs.reshape(num_center_objects, -1)
        _, best_idx = pred_probs_flattened.max(dim=-1)
        controls = self.ctrl_grids.cpu()  # (1, num_motion_modes, 2)
        best_ctrl = controls[best_idx]  # (num_center_objects, 2)
        # print(best_idx, best_ctrl)

        # Gets next state
        controls = best_ctrl[:, None]  # (num_center_objects, 1, 2)
        state_nxt = bicycle_RK4(state_cur, controls, dt=0.1, car_length=car_length)
        return state_nxt[:, 0, [0, 1, 3]].cpu().numpy()

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
        center_objects_feature = center_objects_feature[None, ...].repeat(
            self.num_motion_modes, 1, 1
        )  # (num_motion_modes, num_center_objects, C)

        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid
        obj_feature = obj_feature.permute(1, 0, 2).contiguous()  # (num_objects, num_center_objects, C)

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid
        map_feature = map_feature.permute(1, 0, 2).contiguous()  # (num_polylines, num_center_objects, C)

        # Get positional embedding of the query
        obj_pos_embed = gen_sineembed_for_position(obj_pos.permute(1, 0, 2)[:, :, 0:2], hidden_dim=self.d_model
                                                  ).contiguous()  # (num_objects, num_center_objects, C)

        map_pos_embed = gen_sineembed_for_position(map_pos.permute(1, 0, 2)[:, :, 0:2], hidden_dim=self.d_model
                                                  ).contiguous()  # (num_polylines, num_center_objects, C)

        center_pos_embed = obj_pos_embed[track_index_to_predict,
                                         torch.arange(num_center_objects), :]  # (num_center_objects, C)
        center_pos_embed = center_pos_embed.unsqueeze(0).repeat(
            self.num_motion_modes, 1, 1
        )  # (num_motion_modes, num_center_objects, C)

        # Process the query
        query_embed = self.in_proj_query(self.query.to(center_objects_feature))  # (num_motion_modes, C)
        query_embed = query_embed.unsqueeze(1).repeat(1, num_center_objects, 1)
        query_embed = self.pre_query_fusion_layer(
            torch.cat([center_objects_feature, query_embed], dim=-1)
        )  # (num_motion_modes, num_center_objects, C)

        # Initialize prediction without attention
        prediction = self.prediction_layers[0](query_embed)  # (num_motion_modes, num_center_objects, 1)
        pred_probs = prediction[..., 0].permute(1, 0).reshape(
            num_center_objects, self.num_accels, self.num_steers
        )  # (num_center_objects, num_accels, num_steers)
        pred_list = [pred_probs]

        for i in range(self.num_decoder_layers):
            obj_atten = self.obj_atten_layers[i]
            map_atten = self.map_atten_layers[i]
            query_fuison = self.query_fusion_layers[i]
            pred_layer = self.prediction_layers[i + 1]

            obj_query_embed = obj_atten(
                tgt=query_embed,
                memory=obj_feature,
                tgt_mask=None,
                memory_mask=None,
                tgt_pos=center_pos_embed,
                memory_pos=obj_pos_embed,
                memory_key_padding_mask=~obj_mask,
            )

            map_query_embed = map_atten(
                tgt=query_embed,
                memory=map_feature,
                tgt_mask=None,
                memory_mask=None,
                tgt_pos=center_pos_embed,
                memory_pos=map_pos_embed,
                memory_key_padding_mask=~map_mask,
            )

            query_embed = query_fuison(torch.cat([query_embed, obj_query_embed, map_query_embed], dim=-1))

            prediction = pred_layer(query_embed)
            pred_probs = prediction[..., 0].permute(1, 0).reshape(
                num_center_objects, self.num_accels, self.num_steers
            )  # (num_center_objects, num_accels, num_steers)
            pred_list.append(pred_probs)

        state_cur = torch.zeros((num_center_objects, 4))
        state_cur[:, 2] = torch.linalg.norm(input_dict['center_objects_world'][:, 7:9], dim=-1)
        car_length = input_dict['center_objects_world'][:, 3]
        if 'center_gt' in input_dict:  # Training mode. Otherwise, it is in the inference mode.
            self.forward_ret_dict['pred_list'] = pred_list
            self.forward_ret_dict['center_gt'] = input_dict['center_gt']
            self.forward_ret_dict['state_cur'] = state_cur  # in body frame, so (x, y, h) = (0., 0., 0.)
            self.forward_ret_dict['car_length'] = car_length

        batch_dict['pred_list'] = pred_list
        batch_dict['state_cur'] = state_cur.cpu()
        batch_dict['car_length'] = car_length.cpu()
        return batch_dict
