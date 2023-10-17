from typing import Any, Union, Dict, List, Optional
import numpy as np
from mtr.datasets.waymo.waymo_dataset import WaymoDataset
from .visualization.vis_utils import plot_map, plot_signal, plot_traj_with_time, plot_obj_pose
from mtr.utils import common_utils
import torch

class BatchMTREnv:
    def __init__(self,
        num_envs: int,
        dataset: WaymoDataset,
        random_gen = np.random.default_rng(),
        max_step: int = 200,
    ) -> None:
        
        self.num_envs = num_envs
        self.random_gen = random_gen
        self.max_step = max_step
        self.dataset = dataset
        
        # Create the list of environments
        self.envs_list = [MTREnv(dataset, random_gen, max_step) for _ in range(num_envs)]
                
    def reset(self, reset_bool: np.ndarray = None, no_sdc: bool = False):
        if reset_bool is None: # Reset all the environments
            reset_bool = np.ones(self.num_envs, dtype = np.bool)
        else:
            assert reset_bool.shape == (self.num_envs,)
        
        # if reset_bool.sum() > 0 and self.batch_scene_data is not None:
        batch_list = []
        for env, do_rest in zip(self.envs_list, reset_bool):
            if do_rest:
                batch_list.append(env.reset(no_sdc = no_sdc))
            else:
                batch_list.append(env.scene_data)
        
    @property
    def batch_scene_data(self):
        batch_list = [x.scene_data for x in self.envs_list]
        return self.__collate_batch__(batch_list)
            
    def visualize(self, index: Union[int, List] = None, batch_dict: Dict = None, auto_zoom: int = 20):
        if batch_dict is not None:
            pred_trajs_world = batch_dict['pred_trajs_world']
            
        if index is None:
            index = list(range(self.num_envs))    
        if isinstance(index, int):
            index = [index]
        
        ax_list = []
        fig_list = []
        batch_scene_data = self.batch_scene_data
        for i in index:
            env_mask = batch_scene_data['batch_env_idx'] == i
            pred_trajs_world = None if batch_dict is None else batch_dict['pred_trajs_world'][env_mask]
            pred_scores = None if batch_dict is None else batch_dict['pred_scores'][env_mask].cpu().numpy()
            fig, ax = self.envs_list[i].visualize(pred_trajs = pred_trajs_world, 
                                                  pred_scores = pred_scores,
                                                  auto_zoom = auto_zoom, )
            ax_list.append(ax)
            fig_list.append(fig)
        return fig_list, ax_list
        
    def __collate_batch__(self, batch_list):
        """
        Args:
        batch_list:
            scenario_id: (num_center_objects)
            track_index_to_predict (num_center_objects):

            obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            obj_trajs_mask (num_center_objects, num_objects, num_timestamps):
            map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

            obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
            obj_trajs_last_pos: (num_center_objects, num_objects, 3)
            obj_types: (num_objects)
            obj_ids: (num_objects)

            center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            center_objects_type: (num_center_objects)
            center_objects_id: (num_center_objects)
        """
        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            if key == 'obj_trajs':
                batch_env_idx = np.concatenate([np.ones(x.shape[0])*i for i, x in enumerate(val_list)], axis=0)
                
            if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                'obj_trajs_pos', 'obj_trajs_last_pos']:
                val_list = [torch.from_numpy(x) for x in val_list]
                if 'mask' in key:
                    input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list).bool()
                else:
                    input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list).float()
            elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)

        batch_sample_count = [len(x['track_index_to_predict']) for x in batch_list]
        batch_dict = {
            'batch_size': batch_size, 
            'input_dict': input_dict,
            'batch_sample_count': batch_sample_count,
            'batch_env_idx': batch_env_idx,
            }
        return batch_dict
    
    def step(self, rel_se2: np.ndarray):
        '''
        Input:
            rel_se2: (num_envs, 3)
        '''
        batch_scene_data = self.batch_scene_data
        reset_bool = np.zeros(self.num_envs, dtype = np.bool)
        for i, env in enumerate(self.envs_list):
            env_mask = batch_scene_data['batch_env_idx'] == i
            rel_se2_i = rel_se2[env_mask]
            _, _, reset_bool[i], _ = env.step(rel_se2_i)
        return self.batch_scene_data
    
class MTREnv:
    def __init__(
        self,
        dataset: WaymoDataset,
        random_gen = np.random.default_rng(),
        max_step: int = 200,
    ) -> None:
        
        self.num_scenes = len(dataset)
        self.random_gen = random_gen
        self.dataset = dataset
        self.max_timestamp = max_step
        self.reset()
    
    def reset(self, index: int = None, no_sdc: bool = False, predict_type = 'interested', shift = 0):
        # Index of the scene from Dataset
        if index is None or index >= self.num_scenes:
            self.index = self.random_gen.integers(self.num_scenes)
        else:
            self.index = index
                                       
        # Load the raw data from the dataset
        self.scene_id, self.info = self.dataset.load_info(self.index)
        
        self.current_time_index = self.info['current_time_index']
        self.history_length = self.current_time_index + 1
        self.sdc_track_index = None if no_sdc else self.info['sdc_track_index']
        self.history_timestamps = np.array(self.info['timestamps_seconds'][:self.history_length])
        self.dt = self.history_timestamps[1] - self.history_timestamps[0]

        # Load the GT trajectories
        track_infos = self.info['track_infos']

        self.obj_types = np.array(track_infos['object_type'])
        self.obj_ids = np.array(track_infos['object_id'])
        
        # [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        self.obj_trajs_gt = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        
        if shift > 0:
            obj_trajs_full_shift = np.zeros_like(self.obj_trajs_gt)
            obj_trajs_full_shift[:, :-shift,:] = self.obj_trajs_gt[:, shift:,:]
            self.obj_trajs_gt = obj_trajs_full_shift
            
        self.num_objects = self.obj_trajs_gt.shape[0]
        self.obj_trajs_sim = np.zeros((self.num_objects, self.max_timestamp, 10))
        self.obj_trajs_sim[:, :self.history_length] \
            = self.obj_trajs_gt[:, :self.history_length]
        
        # Get Map information
        self.map_infos= self.info['map_infos']
        if predict_type == 'interested':
            self.get_interested_index()
        elif predict_type == 'moving':
            self.get_moving_index()
        else:
            self.get_all_index()
            
        self.center_objects_id = np.array(track_infos['object_id'])[self.track_index_to_predict]
        self.center_objects_type = np.array(track_infos['object_type'])[self.track_index_to_predict]
        
        self.scene_data = self.extract_scene_data()
        
        return self.scene_data
    
    def get_interested_index(self):
        # Get interested objects
        self.track_index_to_predict = np.array(self.info['tracks_to_predict']['track_index'])
        center_objects_mask = np.zeros(self.obj_trajs_gt.shape[0], dtype = np.bool)
        center_objects_mask[self.track_index_to_predict] = True
        current_valid = self.obj_trajs_gt[:, self.current_time_index, -1]
        center_objects_mask = np.logical_and(center_objects_mask, current_valid)
        self.track_index_to_predict = np.argwhere(center_objects_mask).reshape(-1)
        
    def get_all_index(self):
        current_valid = self.obj_trajs_gt[:, self.current_time_index, -1]
        self.track_index_to_predict = np.argwhere(current_valid).reshape(-1)
        
    def get_moving_index(self):
        current_valid = self.obj_trajs_gt[:, self.current_time_index, -1]
        current_speed = np.linalg.norm(self.obj_trajs_gt[:, self.current_time_index, 7:9], axis = -1)
        current_moving = current_speed > 0.1
        self.track_index_to_predict = np.argwhere(np.logical_and(current_valid, current_moving)).reshape(-1)
        
    def step(self, rel_se2: np.ndarray):
        self.step_gt()
        self.step_control(rel_se2)
        self.current_time_index += 1
        
        self.scene_data = self.extract_scene_data()
        self.reward = self.get_reward()
        self.done = self.check_done()
        # self.info = self.get_info()
        
        return self.scene_data, self.reward, self.done, None
        
    def step_control(self, rel_se2: np.ndarray):
        '''
        Step the simulation with the control input
        action: (num_objects, 3) [dx, dy, dtheta]
        # [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        '''
        # Get the current state
        T_cur = self.state_to_SE2(self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index])
        
        # Get the relative SE2 
        # T_rel = self.exp_map(rel_se2)
        T_rel = self.to_SE2(rel_se2[...,0], rel_se2[...,1], rel_se2[...,2]) 
        # print(T_rel)
        
        # Predict the next state
        T_next =np.einsum('...ij,...jk->...ik', T_cur, T_rel)
        
        x_next, y_next, theta_next = self.SE2_to_state(T_next)
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 0] = x_next
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 1] = y_next
        
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 2] = \
                self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index, 2] # z
            
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 3] = \
                self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index, 3] # dx
            
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 4] = \
                self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index, 4] # dy
                
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 5] = \
            self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index, 5]# dz
            
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 6] = theta_next
        
        # predict the velocity          
        v_body = rel_se2/self.dt
        v_body[..., -1] = 0 # No translation 
        v_world = np.einsum('...ij,...j->...i', T_cur, v_body)
        
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 7] = v_world[..., 0] # vel_x
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 8] = v_world[..., 1] # vel_y
        self.obj_trajs_sim[self.track_index_to_predict, self.current_time_index + 1, 9] = 1
        
    def step_gt(self):
        '''
        Copy the ground truth trajectory to the next time step
        '''
        next_time_index = self.current_time_index + 1
        if next_time_index < self.obj_trajs_gt.shape[1]:
            self.obj_trajs_sim[:, next_time_index] = self.obj_trajs_gt[:, next_time_index]
        
    def check_done(self):
        time_exceed = self.current_time_index >= self.max_timestamp
        return time_exceed
    
    def extract_scene_data(self):
        # We only extract a fixed trailing window of the past trajectory
        end_idx = self.current_time_index + 1
        begin_idx = end_idx - self.history_length
        
        obj_trajs_past = np.copy(self.obj_trajs_sim[:, begin_idx:end_idx])
        obj_trajs_past[:, :-1] = 0
        center_objects = obj_trajs_past[self.track_index_to_predict, -1]
        
        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos,
            track_index_to_predict_new, obj_types, obj_ids) \
        = self.create_agent_data_for_center_objects(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past, 
            track_index_to_predict=self.track_index_to_predict,
            sdc_track_index=self.sdc_track_index,
            timestamps=self.history_timestamps,
            obj_types=self.obj_types,
            obj_ids=self.obj_ids
        )
        
        # Extract the map information
        # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9),
        # (num_center_objects, num_topk_polylines, num_points_each_polyline)
        map_polylines_data, map_polylines_mask, map_polylines_center = self.dataset.create_map_data_for_center_objects(
                center_objects=center_objects, map_infos=self.map_infos,
                center_offset=(30.0, 0), # !Hardcoded
            )   
        
        ret_dict = {
            'scenario_id': np.array([self.scene_id] * len(self.track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': self.center_objects_id,
            'center_objects_type': self.center_objects_type,
            
            'map_polylines': map_polylines_data,
            'map_polylines_mask': map_polylines_mask,
            'map_polylines_center': map_polylines_center,
        }
        
        return ret_dict
    
    def get_reward(self):
        return 0
    
    def get_info(self):
        return None
   
    ######################## State Transformations ########################
    @staticmethod
    def to_SE2(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        T = np.zeros((*x.shape, 3, 3))
        T[..., 0, 0] = np.cos(theta)
        T[..., 0, 1] = -np.sin(theta)
        T[..., 0, 2] = x
        T[..., 1, 0] = np.sin(theta)
        T[..., 1, 1] = np.cos(theta)
        T[..., 1, 2] = y
        T[..., 2, 2] = 1
        
        return T
    
    def SE2_to_state(self, T: np.ndarray) -> np.ndarray:
        theta = np.arctan2(T[..., 1, 0], T[..., 0, 0])
        x = T[..., 0, 2]
        y = T[..., 1, 2]
        return x, y, theta

    def state_to_SE2(self, state: np.ndarray) -> np.ndarray:
        '''
        Input:
            State: (..., 10) [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        Ouput:
            SE2: (..., 3, 3)
        '''
        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 6]
        return self.to_SE2(x, y, theta)
    
    def exp_map(self, se2: np.ndarray) -> np.ndarray:
        '''
        Input:
            se2: (..., 3) [ux, uy, theta]
        Output:
            SE3: (..., 3, 3)
        '''
        ux = se2[..., 0]
        uy = se2[..., 1]
        theta = se2[..., 2]

        cosine = np.cos(theta)
        sine = np.sin(theta)
        
        small_theta = np.abs(theta) < 3e-2
        
        theta2 = theta**2
        theta3 = theta**3
        theta_nz = np.where(small_theta, 1, theta) # theta if theta != 0 else 1
        sine_by_theta = np.where(small_theta, 1 - theta2 / 6, sine / theta_nz)
        cosine_minus_one_by_theta = np.where(
            small_theta, -theta / 2 + theta3 / 24, (cosine - 1) / theta_nz
        )
        
        # Compute the translation
        x = sine_by_theta * ux + cosine_minus_one_by_theta * uy
        y = sine_by_theta * uy - cosine_minus_one_by_theta * ux
        
        return self.to_SE2(x, y, theta)
     
    def log_map(self, SE2: np.ndarray)->np.ndarray:
        '''
        Log map of SE2
        input:
            SE2: (..., 3, 3)
        output:
            se2: (..., 3) [ux, uy, theta] 
        '''
        theta = np.arctan2(SE2[..., 1, 0], SE2[..., 0, 0])
        cosine = np.cos(theta)
        sine = np.sin(theta)
        
        x = SE2[..., 0, 2]
        y = SE2[..., 1, 2]

        # Compute the approximations when theta is near to 0
        small_theta = np.abs(theta) < 3e-2
        sine_nz = np.where(small_theta, 1, sine)
        half_theta_by_tan_half_theta = (
            0.5
            * (1 + cosine)
            * np.where(small_theta, 1 + sine**2 / 6, theta / sine_nz)
        )
        half_theta = 0.5 * theta

        # Compute the translation
        ux = half_theta_by_tan_half_theta * x + half_theta * y
        uy = half_theta_by_tan_half_theta * y - half_theta * x
        return np.stack([ux, uy, theta], axis = -1)
    
    ######################## Process Data ########################
    def create_agent_data_for_center_objects(
            self, 
            center_objects,
            obj_trajs_past,
            track_index_to_predict,
            sdc_track_index,
            timestamps,
            obj_types,
            obj_ids,
        ):
        obj_trajs_data, obj_trajs_mask = self.generate_centered_trajs_for_agents(
            center_objects=center_objects, 
            obj_trajs_past=obj_trajs_past,
            obj_types=obj_types, 
            center_indices=track_index_to_predict,
            sdc_index=sdc_track_index,
            timestamps=timestamps, 
        )

        # filter invalid past trajs
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        # sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

        # generate the final valid position of each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        return (
            obj_trajs_data,
            obj_trajs_mask > 0,
            obj_trajs_pos,
            obj_trajs_last_pos,
            track_index_to_predict_new,
            # sdc_track_index_new,
            obj_types,
            obj_ids
        )
    
    def generate_centered_trajs_for_agents(
            self, 
            center_objects,
            obj_trajs_past,
            obj_types,
            center_indices,
            sdc_index,
            timestamps
        ):
        """
        This function create 'agent-centric' trajectories for each object around each center objects
        Args:
            center_objects (num_center_objects, 10): the current states of objects that need to be predicted
                [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid], 
            obj_trajs_past (num_objects, num_timestamps, 10): the past trajectories of all objects
            obj_types: List[str] (num_objects): the type of each object
            center_indices (num_center_objects): the index of center objects in obj_trajs_past
            sdc_index (int): the index of the self-driving car in obj_trajs_past
            timestamps ([float]): list of timestamps for the past trajectories
            
        Returns:
            ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
        """
        assert obj_trajs_past.shape[-1] == 10
        assert center_objects.shape[-1] == 10
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, _ = obj_trajs_past.shape
        
        # transform the coordinate systems to each centered objects
        # [num_center_objects, num_objects, num_timestamps, num_attrs]
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        ## generate the attributes for each object
        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
        object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
        object_onehot_mask[np.arange(num_center_objects), center_indices, :, 3] = 1
        if sdc_index is not None:
            object_onehot_mask[:, sdc_index, :, 4] = 1

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, np.arange(num_timestamps), np.arange(num_timestamps)] = 1
        object_time_embedding[:, :, np.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        acce[:, :, 0, :] = acce[:, :, 1, :]

        ret_obj_trajs = np.concatenate((
            obj_trajs[:, :, :, 0:6], 
            object_onehot_mask,
            object_time_embedding, 
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9], 
            acce,
        ), axis=-1)

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        return ret_obj_trajs, ret_obj_valid_mask
    
    def transform_trajs_to_center_coords(
        self, 
        obj_trajs,
        center_xyz,
        center_heading,
        heading_index,
        rot_vel_index=None,
    ):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        return:
            obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = np.copy(obj_trajs)[None].repeat(num_center_objects, axis=0)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = self.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = self.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
                angle=-center_heading
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs
    
    @staticmethod
    def rotate_points_along_z(points, angle):
            """
            Args:
                points: (B, N, 3 + C)
                angle: (B), angle along z-axis, angle increases x ==> y
            Returns:

            """
            cosa = np.cos(angle)
            sina = np.sin(angle)
            zeros = np.zeros(points.shape[0])
            if points.shape[-1] == 2:
                rot_matrix = np.stack((
                    cosa,  sina,
                    -sina, cosa
                ), axis=1).reshape(-1, 2, 2)
                points_rot = np.matmul(points, rot_matrix)
            else:
                ones = np.ones(points.shape[0])
                rot_matrix = np.stack((
                    cosa,  sina, zeros,
                    -sina, cosa, zeros,
                    zeros, zeros, ones
                ), axis=1).reshape(-1, 3, 3)
                points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
                points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
            return points_rot
        
    ######################## Visualization ########################
    def visualize(self, 
        pred_trajs: np.ndarray = None,
        pred_scores: np.ndarray = None,          
        auto_zoom: int = 20,
    ):
        fig, ax = plot_map(self.map_infos)
        # plot_signal(self.info['dynamic_map_infos'], self.current_time_index, ax)
        plot_traj_with_time(
            self.center_objects_type, 
            self.obj_trajs_sim[self.track_index_to_predict, :self.current_time_index + 1],
            [i*self.dt for i in range(self.current_time_index + 1)],
            ax=ax,
            fig=fig,)
        for obj_type, traj in zip(
            self.obj_types, self.obj_trajs_sim[:, self.current_time_index]
        ):
            plot_obj_pose(obj_type, traj, ax=ax)

        if pred_trajs is not None:
            for i in range(pred_trajs.shape[0]):
                traj = pred_trajs[i]
                score = pred_scores[i]
                for future, score in zip(traj, score):
                    if score < 0.1:
                        continue
                    ax.plot(future[:, 0], future[:, 1],
                            color='xkcd:russet', linewidth=2, linestyle='-', 
                            alpha=score*0.7+0.3, zorder=2)

        if auto_zoom>=0:
            # Zoom in to the current scene
            valid_traj_mask = self.obj_trajs_sim[..., -1] > 0
            valid_traj = self.obj_trajs_sim[valid_traj_mask]
            max_x = np.max(valid_traj[..., 0])+auto_zoom
            min_x = np.min(valid_traj[..., 0])-auto_zoom
            max_y = np.max(valid_traj[..., 1])+auto_zoom
            min_y = np.min(valid_traj[..., 1])-auto_zoom
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
                        
        ax.set_title(f'Scene {self.scene_id} at {self.current_time_index/10} seconds')
        return fig, ax