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
from mtr.models.motion_decoder.bc_decoder import BCDecoder
from typing import List, Dict


_DEFAULT_CONTROL_FUNC = lambda state: state.object_metadata.is_modeled
    
@actor_core.register_actor_core
class SimAgentMTR(actor_core.WaymaxActorCore):
    def __init__(
        self,
        context_encoder: MTREncoder,
        motion_decoder: BCDecoder,
        is_controlled_func: Optional[
            Callable[[datatypes.SimulatorState], jax.Array]
        ]= None,  
        history_length: int = 11, 
        dt: float = 0.1,
    ):
        super().__init__()
        if is_controlled_func is None:
            is_controlled_func = _DEFAULT_CONTROL_FUNC
        self.is_controlled_func = is_controlled_func        
        self.history_length = history_length
        self.timestamps = np.arange(self.history_length) * dt
        
        # Model related
        self.context_encoder = context_encoder
        self.motion_decoder = motion_decoder
       
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
        with torch.no_grad():
            encoded_state, is_controlled = self.encoding_state(state)
            output = self.forward_decoder(encoded_state)
        
        actions_sampled = self.sample(output)['sample'].detach().cpu().numpy()
        
        return self.sample_to_action(actions_sampled, is_controlled)
    
    def encoding_state(self, state: datatypes.SimulatorState, is_controlled: jax.Array = None):
        if is_controlled is None:
            is_controlled = self.is_controlled_func(state)
        # update the state
        state.object_metadata.is_modeled = is_controlled
        state.object_metadata.is_controlled = is_controlled
        input_dict = process_input(state, is_controlled)
        input_dict_batch = encoder_collate_batch([input_dict])
        
        encoded_state = self.forward_encoder(input_dict_batch)
        
        return encoded_state, is_controlled
        
    def sample_to_action(self, sample: np.ndarray, is_controlled: jax.Array)->datatypes.Action:
        """Converts a sample to an waymax action."""
        actions_array = np.zeros((is_controlled.shape[0], 3))
        actions_array[is_controlled] = sample
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
    
    ##### Forward functions #####
    def forward_encoder(self, batch_dict):
        """
        Forward pass through the context encoder.

        Args:
            batch_dict (dict): The batch dictionary.

        Returns:
            encoder_dict: The batch dictionary with the context encoding added.
        """
        encoder_dict = self.context_encoder(batch_dict, retain_input = False)
        return encoder_dict
    
    def forward_decoder(self, encoder_dict: Dict):
        """
        Forward pass through the motion decoder.

        Args:
            encoder_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the motion prediction.
        """ 
        return self.motion_decoder(encoder_dict)
    
    def forward(self, batch_dict):
        """
        Forward pass through the context encoder and motion decoder.

        Args:
            batch_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the context encoding and motion prediction.
        """
        encoder_dict = self.forward_encoder(batch_dict)
        output_dict = self.forward_decoder(encoder_dict)
        return output_dict

    def sample(self, output_dict):
        """
        Sample a trajectory from the motion decoder.

        Args:
            batch_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the sampled trajectory added.
        """
    
        
        pred_ctrls, pred_scores = output_dict['pred_list'][-1]
        mode, mix, gmm = self.motion_decoder.build_gmm_distribution(pred_ctrls, pred_scores)
        
        # sample_action = gmm.sample()
        mode: torch.distributions.MultivariateNormal
        mix: torch.distributions.Categorical
        
        # Sample from all Gaussian
        sample_all = mode.rsample() # [Batch, M, 3]
        sample_all_log_prob = mode.log_prob(sample_all)
        
        sample_mode = mix.sample() # [Batch]
        sample_mode_log_prob = mix.log_prob(sample_mode)
        
        sample_action = torch.gather(
            sample_all, 
            1, 
            sample_mode.unsqueeze(-1).unsqueeze(-1).repeat_interleave(sample_all.shape[-1], dim=-1)
        ).squeeze(-2)
        
        sample_action_log_prob = torch.gather(
            sample_all_log_prob, 
            1, 
            sample_mode.unsqueeze(-1)
        ).squeeze(-1)  + sample_mode_log_prob
                    
        sample = sample_action * self.motion_decoder.output_std + self.motion_decoder.output_mean
        
        sample_dict = {
            'sample': sample,
            'log_prob': sample_action_log_prob
        }
            
        return sample_dict
    
    def sample_best(self, output_dict):
        """
        Sample a trajectory from the motion decoder.

        Args:
            batch_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the sampled trajectory added.
        """
    
        cur_decoder = self.motion_decoder
        
        pred_ctrls, pred_scores = output_dict['pred_list'][-1]
        
        best_idx = torch.argmax(pred_scores, dim=-1)
        
        # take value from the best index
        sample = pred_ctrls[torch.arange(pred_ctrls.shape[0]), best_idx, :3]
        
        sample = sample * cur_decoder.output_std + cur_decoder.output_mean
        
        sample_dict = {'sample': sample}
        
        return sample_dict
    
    ##### Utility functions #####
    def eval(self):
        self.context_encoder.eval()
        self.motion_decoder.eval()
    
    def train(self):
        self.context_encoder.train()
        self.motion_decoder.train()
        
    def to(self, device: torch.device):
        self.context_encoder.to(device)
        self.motion_decoder.to(device)
        
    def cuda(self):
        self.to(torch.device('cuda'))