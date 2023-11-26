from typing import Callable, Optional
import jax
import jax.numpy as jnp
from waymax import agents, datatypes
from waymax.agents import actor_core

from rl_env.env_utils import *

from typing import Dict, Tuple
from tools.mtr_lightning import MTR_Lightning
from rl.joint_policy import JointPolicy


_DEFAULT_CONTROL_FUNC = lambda state: state.object_metadata.is_modeled

@actor_core.register_actor_core
class SimAgentMTR(actor_core.WaymaxActorCore):
    def __init__(
        self,
        model_config,
        model_path: str,
        history_length: int = 11, 
        dt: float = 0.1,
        is_controlled_func: Optional[
            Callable[[datatypes.SimulatorState], jax.Array]
        ]= None,  
    ):
        super().__init__()
        if is_controlled_func is None:
            is_controlled_func = _DEFAULT_CONTROL_FUNC
        self.is_controlled_func = is_controlled_func
        
        # self.model = MTR_Lightning.load_from_checkpoint(model_path).cuda()
        self.model = JointPolicy(model_config)
        self.model.load_params_from_file(model_path)
        self.model.to('cuda')
        
        self.history_length = history_length
        self.timestamps = np.arange(self.history_length) * dt
       
    def init(self, rng: jax.Array, state: datatypes.SimulatorState):
        """Returns an empty initial state."""
        raise NotImplementedError
    
    def select_action(
        self,
        params: actor_core.Params,
        state: datatypes.SimulatorState,
        actor_state: actor_core.ActorState,
        rng: jax.Array,
    ) -> agents.WaymaxActorOutput:
        """Selects an action given the current simulator state."""
        
        # actor_type = actor_state['actor_type']
        # del params, actor_state, rng
        
        is_controlled = self.is_controlled_func(state)
        input_dict = self.process_input(state, is_controlled)
        input_dict_batch = collate_batch([input_dict])
        
        # Do a forward pass
        # ! hard code
        batch_decoder_mapping = {
            'agent': [i for i in range(is_controlled.sum())]
        }
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_dict_batch, batch_decoder_mapping)
        
        actions_sampled = self.model.sample(output)['agent']['sample'].detach().cpu().numpy()
        
        actions_array = np.zeros((is_controlled.shape[0], 3))
        actions_array[is_controlled] = actions_sampled
        actions_valid = jnp.asarray(is_controlled[...,None])
        
        actions = datatypes.Action(data=jnp.asarray(actions_array), valid=actions_valid)
        
        return actor_core.WaymaxActorOutput(
            action=actions,
            actor_state=None,
            is_controlled=is_controlled,
        )
            
    @property
    def name(self) -> str:
        return 'mtr'
        
    def process_input(
        self, scenario: datatypes.SimulatorState,
        is_controlled: np.ndarray
    ) -> Dict:
        """
        Extracts scene data from a simulator state.

        Args:
            scenario (datatypes.SimulatorState): The simulator state.

        Returns:
            Dict: A dictionary containing the extracted scene data.
        """
        # Extract Objects Meta Data
        obj_metadata: datatypes.ObjectMetadata = scenario.object_metadata
        obj_ids = obj_metadata.ids
        obj_types = self._convert_obj_type(obj_metadata)
        sdc_track_index = np.where(obj_metadata.is_sdc)[0][0] # only one sdc
        track_index_to_predict = np.where(is_controlled)[0]
        current_time_index = scenario.timestep
        # print(track_index_to_predict)
        # Extract Objects Trajectory
        trajectory: datatypes.Trajectory = scenario.sim_trajectory # TODO: check if this is the right trajectory

        # Extract Objects State
        obj_trajs = self._stack_traj(trajectory)
        
        end_index = current_time_index + 1
        start_index = max(0, end_index - self.history_length)
        obj_trajs_past = obj_trajs[:, start_index:end_index, :]
        center_objects = obj_trajs_past[track_index_to_predict, -1]
        center_objects_id = obj_ids[track_index_to_predict]
        center_objects_type = obj_types[track_index_to_predict]
        
        if obj_trajs_past.shape[1] < self.history_length:
            # pad with zeros
            obj_trajs_past = np.pad(obj_trajs_past, ((0, 0), (self.history_length - obj_trajs_past.shape[1], 0), (0, 0))) 

        # create agent centric trajectory
        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos,
            track_index_to_predict_new, obj_types, obj_ids) \
        = create_agent_data_for_center_objects(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past, 
            track_index_to_predict=track_index_to_predict,
            sdc_track_index=sdc_track_index,
            timestamps=self.timestamps,
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
    
    
    
    