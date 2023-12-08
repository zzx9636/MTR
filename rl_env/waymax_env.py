# This class is a wrapper of Waymax to simulate the environment from WOMD

import numpy as np
import jax
from jax import jit
from jax import numpy as jnp

import mediapy
from tqdm import tqdm
import dataclasses

from waymax import config as _config
from waymax import datatypes  
from waymax import dynamics
from waymax import env as waymax_env
from waymax import agents
from waymax.agents import actor_core

from typing import Dict, Tuple, Union, List

from waymax.env import typedefs as types

from rl_env.env_utils import *
from rl_env.rewards.ReachAvoidMetrics import ReachAvoidMetrics
from tools.mtr_lightning import MTR_Lightning
import tensorflow as tf
 
class MultiAgentEnvironment(waymax_env.BaseEnvironment):
    def __init__(
            self,
            dynamics_model: dynamics.DynamicsModel,
            config: _config.EnvironmentConfig,
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
        self._reward_function = ReachAvoidMetrics(config.rewards)
        self._dynamics_model = dynamics_model
        self.config = config
        
        # create a GT actor for agent. Sim Agent will be filtered out during step as a post-processing step
        self.gt_actor = agents.create_expert_actor( 
            dynamics_model=dynamics_model,
            is_controlled_func=lambda state: state.object_metadata.is_valid)
        
        # Useful jited functions 
        self.jit_step = jit(self.step)
        self.jit_gt_action = jit(self.gt_actor.select_action)
        
    def step_sim_agent(
        self,
        current_state: datatypes.SimulatorState,
        sim_agent_action_list: List[datatypes.Action]
    ) -> datatypes.SimulatorState:
        """
        Steps the simulation agent.
        """
        # Step the GT Policy
        gt_action_full: datatypes.Action = self.jit_gt_action({}, current_state, None, None)
        
        # do a validation check
        is_controlled_stack = jnp.vstack([action.is_controlled for action in sim_agent_action_list])
        num_controlled = jnp.sum(is_controlled_stack, axis=0) # (num_agent, 1)
        if jnp.any(num_controlled > 1):
            raise Warning("An agent is controlled by more than one policy")
        use_log = num_controlled == 0
        
        gt_action = actor_core.WaymaxActorOutput(
            action = gt_action_full.action,
            actor_state = None,
            is_controlled = use_log
        )
        
        # merge the action
        sim_agent_action_list.append(gt_action)
        action_merged = agents.merge_actions(sim_agent_action_list)
        
        next_state = self.jit_step(current_state, action_merged)
        
        return next_state
        
    def metrics(self, state: datatypes.SimulatorState, action: datatypes.Action):
        '''
        Compute the metrics for the current state and action
        '''
        return self._reward_function.compute(state)
    
    
                
        
        
    
        
    