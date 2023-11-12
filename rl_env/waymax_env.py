# This class is a wrapper of Waymax to simulate the environment from WOMD

import numpy as np
import jax
from jax import jit
from jax import numpy as jnp

import mediapy
from tqdm import tqdm
import dataclasses

from waymax import config as waymax_config
from waymax import dataloader
from waymax import datatypes  
from waymax import visualization

from typing import Dict

from rl_env.env_utils import *

class WaymaxEnv:
    def __init__(self):
        self.current_time_index = 10
    
    def extract_scene_data(self, scenario: datatypes.SimulatorState, history_length: int = 11)->Dict:
        """Extracts scene data from a simulator state."""
        # Extract Objects Meta Data
        obj_metadata: datatypes.ObjectMetadata = scenario.object_metadata
        obj_ids = obj_metadata.ids
        obj_types = self._convert_obj_type(obj_metadata)
        sdc_track_index = np.where(obj_metadata.is_sdc)[0][0] # only one sdc
        track_index_to_predict = np.where(obj_metadata.is_modeled)[0]

        # Extract Objects Trajectory
        trajectory: datatypes.Trajectory = scenario.log_trajectory # TODO: check if this is the right trajectory
        timestamps = trajectory.timestamp_micros[0, :history_length]/1e6
        dt = timestamps[1] - timestamps[0]
        
        # Extract Objects State
        obj_trajs = self._stack_traj(trajectory)
        
        end_index = self.current_time_index + 1
        start_index = max(0, end_index - history_length)
        obj_trajs_past = obj_trajs[:, start_index:end_index, :]
        center_objects = obj_trajs_past[track_index_to_predict, -1]
        center_objects_id = obj_ids[track_index_to_predict]
        center_objects_type = obj_types[track_index_to_predict]
        
        if obj_trajs_past.shape[1] < history_length:
            # pad with zeros
            obj_trajs_past = np.pad(obj_trajs_past, ((0, 0), (history_length - obj_trajs_past.shape[1], 0), (0, 0))) 

        # create agent centric trajectory
        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos,
            track_index_to_predict_new, obj_types, obj_ids) \
        = create_agent_data_for_center_objects(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past, 
            track_index_to_predict=track_index_to_predict,
            sdc_track_index=sdc_track_index,
            timestamps=timestamps,
            obj_types=obj_types,
            obj_ids=obj_ids
        )
        
        polylines = self._stack_map(scenario.roadgraph_points)
        # Extract the map information
        # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9),
        # (num_center_objects, num_topk_polylines, num_points_each_polyline)
        map_polylines_data, map_polylines_mask, map_polylines_center = create_map_data_for_center_objects(
                center_objects=center_objects, 
                polylines=polylines,
                center_offset=(0, 0), # !Hardcoded
            )   
        
        ret_dict = {
            # 'scenario_id': np.array([self.scene_id] * len(track_index_to_predict_new)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': center_objects_id,
            'center_objects_type': center_objects_type,
            
            'map_polylines': map_polylines_data,
            'map_polylines_mask': map_polylines_mask,
            'map_polylines_center': map_polylines_center,
        }
        
        return ret_dict
               
    #################### Utils ####################
    @staticmethod
    def _stack_traj(traj: datatypes.Trajectory) -> np.ndarray:
        """Stacks a trajectory into a 10D array."""
        # [cx, cy, cz, length, width, height, heading, vel_x, vel_y, valid]
        return np.stack([
            traj.x,
            traj.y,
            traj.z,
            traj.length,
            traj.width,
            traj.height,
            traj.yaw,
            traj.vel_x,
            traj.vel_y,
            traj.valid,
        ], axis=-1) # [num_tracks, num_steps, 10]
        
    @staticmethod
    def _stack_map(map_data: datatypes.RoadgraphPoints) -> np.ndarray:
        return np.stack([
            map_data.x,
            map_data.y,
            map_data.z,
            map_data.dir_x,
            map_data.dir_y,
            map_data.dir_z,
            map_data.types,
            # map_data.ids,
            # map_data.valid,
        ], axis=-1) # [num_tracks, num_steps, 9]
    
    @staticmethod
    def _convert_obj_type(obj_metadata: datatypes.ObjectMetadata)->np.ndarray:
        """Converts object type from int to string"""
        
        str_map = ['TYPE_UNSET', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST', 'TYPE_OTHER']
        
        obj_types_int = obj_metadata.object_types
        obj_types_str = np.array([str_map[i] for i in obj_types_int])
        
        return obj_types_str