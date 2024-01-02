import numpy as np
import torch
from mtr.utils import common_utils
from waymax import datatypes
from waymax.agents import actor_core
from waymax import dataloader
from waymax import config as waymax_config
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import numpy as np
import jax
from jax import numpy as jnp
# from rl_env.env_utils import *
from typing import Dict, Tuple, List
from torch.utils.data import IterableDataset
from waymax.dynamics import bicycle_model
    
def create_iter(data_config: waymax_config.DatasetConfig)-> iter(Tuple[str, datatypes.SimulatorState]):
    # Write a custom dataloader that loads scenario IDs.
    def _preprocess(serialized: bytes) -> dict[str, tf.Tensor]:
        womd_features = dataloader.womd_utils.get_features_description(
            include_sdc_paths=data_config.include_sdc_paths,
            max_num_rg_points=data_config.max_num_rg_points,
            num_paths=data_config.num_paths,
            num_points_per_path=data_config.num_points_per_path,
        )
        womd_features['scenario/id'] = tf.io.FixedLenFeature([1], tf.string)

        deserialized = tf.io.parse_example(serialized, womd_features)
        parsed_id = deserialized.pop('scenario/id')
        deserialized['scenario/id'] = tf.io.decode_raw(parsed_id, tf.uint8)
        return dataloader.preprocess_womd_example(
            deserialized,
            aggregate_timesteps=data_config.aggregate_timesteps,
            max_num_objects=data_config.max_num_objects,
        )
        
    def _postprocess(example: dict[str, tf.Tensor]):
        scenario = dataloader.simulator_state_from_womd_dict(example)
        scenario_id = example['scenario/id']
        return scenario_id, scenario
    
    def decode_bytes(data_iter):
        # Force use CPU
        # with tf.device('/cpu:0'):
        for scenario_id, scenario in data_iter:
            scenario_id = scenario_id.tobytes().decode('utf-8')
            yield scenario_id, scenario
                
    return decode_bytes(dataloader.get_data_generator(
            data_config, _preprocess, _postprocess
        ))
        
class WomdLoader:
    def __init__(self, data_config: waymax_config.DatasetConfig) -> None:
        self.data_config = data_config
        self.reset()
        
    def reset(self):
        self.iter = create_iter(self.data_config)
    
    def next(self):
        return next(self.iter)
    
def sample_to_action(sample: np.ndarray, is_controlled: jax.Array)->datatypes.Action:
    """Converts a action [dx, dy, dyaw] to an waymax action."""
    action_dim = sample.shape[-1]
    actions_array = np.zeros((is_controlled.shape[0], action_dim))
    actions_array[is_controlled] = sample
    actions_valid = jnp.asarray(is_controlled[...,None])
    
    actions = datatypes.Action(data=jnp.asarray(actions_array), valid=actions_valid)
    
    return actor_core.WaymaxActorOutput(
        action=actions,
        actor_state=None,
        is_controlled=is_controlled,
    )