# This class is a wrapper of Waymax to simulate the environment from WOMD

import numpy as np
import jax
from jax import jit
from jax import numpy as jnp

import mediapy
from tqdm import tqdm
import dataclasses

from waymax import config as waymax_config, config as _config, dynamics as _dynamics
from waymax import dataloader
from waymax import datatypes  
from waymax import visualization
from waymax import dynamics
from waymax import env as waymax_env
from waymax import agents

from typing import Dict, Tuple, Union

from waymax.env import typedefs as types

from rl_env.env_utils import *
from rl_env.rewards.dictionary_reward import DictionaryReward
from tools.mtr_lightning import MTR_Lightning
import tensorflow as tf

DEFAULT_AGENT_FILTER = lambda state: state.object_metadata.is_modeled

def womd_loader(data_config: waymax_config.DatasetConfig)-> iter(Tuple[str, datatypes.SimulatorState]):
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
        # print(deserialized['scenario/id'].tobytes())
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
        for scenario_id, scenario in data_iter:
            scenario_id = scenario_id.tobytes().decode('utf-8')
            yield scenario_id, scenario
            
    return decode_bytes(dataloader.get_data_generator(
            data_config, _preprocess, _postprocess
        ))
 
class MultiAgentEnvironment(waymax_env.BaseEnvironment):
    def __init__(
            self,
            dynamics_model: dynamics.DynamicsModel,
            config: _config.EnvironmentConfig,
            agent_filter=DEFAULT_AGENT_FILTER,
        ):
        """
        Initializes a new instance of the WaymaxEnv class.

        Args:
            dynamics_model (dynamics.DynamicsModel): The dynamics model used for simulating the environment.
            config (_config.EnvironmentConfig): The configuration object for the environment.
        """
        # ! Do not call super().__init__ here, it will cause error
        # super().__init__(dynamics_model, config)
        
        # override the reward function with the dictionary reward function
        self._reward_function = DictionaryReward(config.rewards)
        self._dynamics_model = dynamics_model
        self.config = config
        
        self.gt_actor = agents.create_expert_actor(
            dynamics_model=dynamics_model,
            is_controlled_func=lambda state: agent_filter(state),
        )
        
        # Useful jited functions
        self.jit_step = jit(self.step)
        self.jit_gt_action = jit(self.gt_actor.select_action)
        
    def step_sim_agent(
        current_state: datatypes.SimulatorState,
        sim_agent_action: Union[np.ndarray, jax.Array],
        
        ) -> datatypes.SimulatorState:
        """
        Steps the simulation agent.
        """
        pass
        
    
        
    