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
from waymax import dynamics
from waymax import env as waymax_env
from waymax import agents

from typing import Dict, Tuple

from rl_env.env_utils import *
from tools.mtr_lightning import MTR_Lightning

class WaymaxEnv:
    def __init__(self, config):
        
        # Load the config
        self.config = config
        self.init_steps = self.config.get('INIT_STEPS', 11)
        self.history_length = self.config.get('HISTORY_LENGTH', 11)
        self.max_num_objects = self.config.get('MAX_NUM_OBJECTS', 32)
        self.dt = self.config.get('DT', 0.1)
        self.dynamics_type = self.config.get('DYNAMICS_TYPE', 'DeltaLocal')
        self.use_jit = self.config.get('USE_JIT', True)
        self.timestamps = np.arange(self.history_length) * self.dt
        
        
        if self.dynamics_type == 'DeltaLocal':
            dynamics_model = dynamics.DeltaLocal()
        elif self.dynamics_type == 'DeltaGlobal':
            dynamics_model = dynamics.DeltaGlobal()
        elif self.dynamics_type == 'Bicycle':
            dynamics_model = dynamics.InvertibleBicycleModel()
        elif self.dynamics_type == 'StateDynamics':
            dynamics_model = dynamics.StateDynamics()
        else:
            raise ValueError(f'Unknown dynamics type: {self.dynamics_type}')
            
        
        # Construct the simulator
        # TODO: This controlled_object is related to how reward is calculated, so we need to explicitly figure it out
        self.env = waymax_env.MultiAgentEnvironment(
            dynamics_model=dynamics_model,
            config=dataclasses.replace(
                waymax_config.EnvironmentConfig(),
                max_num_objects=self.max_num_objects,
                init_steps = self.init_steps,
                allow_new_objects_after_warmup=True,
                controlled_object=waymax_config.ObjectType.VALID,
            ),
        )
        
        # Jit-ify the simulator
        if self.use_jit:
            self.env_step = jit(self.env.step)
        else:
            self.env_step = self.env.step
            
        # Load Model
        self.model = MTR_Lightning(self.config)
        
    def reset(self, scenario: datatypes.SimulatorState) -> Tuple[datatypes.SimulatorState, Dict]:
        """
        Resets the environment to the initial state and returns the initial state and input dictionary.

        Args:
            scenario (datatypes.SimulatorState): The initial state of the simulator.

        Returns:
            Tuple[datatypes.SimulatorState, Dict]: A tuple containing the initial state of the simulator and the input
            dictionary.
        """
        simulation_state: datatypes.SimulatorState = self.env.reset(scenario)
        input_dict: dict = self.process_input(simulation_state)
        
        return simulation_state, input_dict

    def step(self, prev_state: datatypes.SimulatorState, action: datatypes.Action) -> Tuple[datatypes.SimulatorState, Dict]:
        """
        Steps the environment forward using the given action and returns the next state and input dictionary.

        Args:
            prev_state (datatypes.SimulatorState): The previous state of the simulator.
            action (np.ndarray): The action to take.

        Returns:
            Tuple[datatypes.SimulatorState, Dict]: A tuple containing the next state of the simulator and the input
            dictionary.
        """
        simulation_state: datatypes.SimulatorState = self.env.step(prev_state, action)
        input_dict: dict = self.process_input(simulation_state)
        
        return simulation_state, input_dict
    
    def observe(self, scenario: datatypes.SimulatorState) -> Tuple[datatypes.SimulatorState, Dict]:
        """
        Observe the current state of the environment.

        Args:
            scenario (datatypes.SimulatorState): The current state of the simulator.

        Returns:
            Tuple[datatypes.SimulatorState, Dict]: A tuple containing the current state of the simulator and a dictionary of processed inputs.
        """
        input_dict: dict = self.process_input(scenario)
        
        return scenario, input_dict
        
   
        """Converts object type from int to string"""
        
        str_map = ['TYPE_UNSET', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST', 'TYPE_OTHER']
        
        obj_types_int = obj_metadata.object_types
        obj_types_str = np.array([str_map[i] for i in obj_types_int])
        
        return obj_types_str