from typing import Callable, Optional
import jax
import jax.numpy as jnp
from waymax import agents, datatypes
from waymax.agents import actor_core

from rl_env.env_utils import *

import numpy as np
import os
import torch
import torch.nn as nn
from mtr.models.context_encoder.mtr_encoder import MTREncoder 
from typing import List, Dict


_DEFAULT_CONTROL_FUNC = lambda state: state.object_metadata.is_modeled
    
class Encoder(nn.Module):
    def __init__(
        self,
        model_cfg,
        is_controlled_func: Optional[
            Callable[[datatypes.SimulatorState], jax.Array]
        ]= None,  
        history_length: int = 11, 
        dt: float = 0.1,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        if is_controlled_func is None:
            is_controlled_func = _DEFAULT_CONTROL_FUNC
        self.is_controlled_func = is_controlled_func        
        self.history_length = history_length
        self.timestamps = np.arange(self.history_length) * dt
        
        # Model related
        self.context_encoder = MTREncoder(model_cfg)
        self.to(device)
               
    def forward(self, state: datatypes.SimulatorState, is_controlled: jax.Array = None):
        if is_controlled is None:
            is_controlled = self.is_controlled_func(state)
        # update the state
        state.object_metadata.is_modeled = is_controlled
        state.object_metadata.is_controlled = is_controlled
        input_dict = process_input(state, is_controlled)
        input_dict_batch = encoder_collate_batch([input_dict])
        
        encoded_state = self.context_encoder(input_dict_batch, retain_input = False)
        
        return encoded_state, is_controlled
        
    def load_model(
        self,
        state_dict: dict
    ):
        model_keys = self.state_dict().keys()  
        state_dict_filtered = {}
        # search for the weights in the state_dict_to_load and save to a new dict
        for key in model_keys:
            for state_key in state_dict.keys():
                if state_key.endswith(key):
                    state_dict_filtered[key] = state_dict[state_key]
                    break
                
        # load the filtered state_dict
        self.load_state_dict(state_dict_filtered, strict=True)