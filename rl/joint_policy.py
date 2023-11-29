# Encoder Decoder Style 

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtr.models.context_encoder.mtr_encoder import MTREncoder 
from mtr.models.motion_decoder.bc_decoder import BCDecoder
from typing import List, Dict

class JointPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config
        
        self.decoder_name_list = config.DECODER_LIST # name of the decoder

        self.context_encoder: nn.Module = MTREncoder(self.model_cfg.CONTEXT_ENCODER)
        self.motion_decoder_list: List[nn.Module] = [BCDecoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER
        ) for _ in self.decoder_name_list]

    def load_params_from_file(self, filename, to_cpu=True):
        """
        Loads parameters from a checkpoint file and assigns them to the context_encoder and motion_decoder models.

        Args:
            filename (str): The path to the checkpoint file.
            to_cpu (bool, optional): Whether to load the parameters to CPU. Defaults to False.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
            
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        
        model_state_disk = checkpoint['state_dict']

        version = checkpoint.get("version", None)
        print('==> Checkpoint trained from version: %s' % version)
        print(f'The number of disk ckpt keys: {len(model_state_disk)}')
        
        # Extract the encoder and decoder state dicts separately
        encoder_state = {}
        decoder_state = {}
        for k, v in model_state_disk.items():
            if 'context_encoder' in k:
                new_key = k.replace('context_encoder.', '').replace('model.', '')
                encoder_state[new_key] = v
            elif 'motion_decoder' in k:
                new_key = k.replace('motion_decoder.', '').replace('model.', '')
                decoder_state[new_key] = v
            else:
                raise ValueError(f'Unexpected key: {k}')
        
        # Load encoder and decoder state dicts
        self.context_encoder.load_state_dict(encoder_state, strict=True)
        for decoder in self.motion_decoder_list:
            decoder.load_state_dict(decoder_state, strict=True)
            
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
    
    def forward_decoder(self, encoder_dict: Dict, batch_decoder_mapping: Dict = None):
        """
        Forward pass through the motion decoder.

        Args:
            encoder_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the motion prediction added.
        """
        output_dict = {}
        for name, decoder in zip(self.decoder_name_list, self.motion_decoder_list):
            # Extract the batch data for the current decoder
            if batch_decoder_mapping is not None:
                idx = batch_decoder_mapping[name]
                
                decoder_input_dict = {
                    'track_index_to_predict': encoder_dict['track_index_to_predict'][idx],
                    'obj_feature': encoder_dict['obj_feature'][idx],
                    'obj_pos': encoder_dict['obj_pos'][idx],
                    'obj_mask': encoder_dict['obj_mask'][idx],
                    'map_feature': encoder_dict['map_feature'][idx],
                    'map_pos': encoder_dict['map_pos'][idx],
                    'map_mask': encoder_dict['map_mask'][idx],
                }
            else:
                decoder_input_dict = encoder_dict
            output_dict[name] = decoder(decoder_input_dict)
        return output_dict
    
    def forward(self, batch_dict, batch_decoder_mapping):
        """
        Forward pass through the context encoder and motion decoder.

        Args:
            batch_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the context encoding and motion prediction added.
        """
        encoder_dict = self.forward_encoder(batch_dict)
        output_dict = self.forward_decoder(encoder_dict, batch_decoder_mapping)
        return output_dict

    def sample(self, output_dict):
        """
        Sample a trajectory from the motion decoder.

        Args:
            batch_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the sampled trajectory added.
        """
        sample_dict = {}
        for i, (name, batch_dict) in enumerate(output_dict.items()):
            cur_decoder = self.motion_decoder_list[i]
            
            pred_ctrls, pred_scores = batch_dict['pred_list'][-1]
            mode, mix, gmm = cur_decoder.build_gmm_distribution(pred_ctrls, pred_scores)
            
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
                        
            sample = sample_action * cur_decoder.output_std + cur_decoder.output_mean
            
            cur_sample = {
                'sample': sample,
                'log_prob': sample_action_log_prob
            }
            
            sample_dict[name] = cur_sample
            
        return sample_dict
    
    def sample_best(self, output_dict):
        """
        Sample a trajectory from the motion decoder.

        Args:
            batch_dict (dict): The batch dictionary.

        Returns:
            output_dict: The batch dictionary with the sampled trajectory added.
        """
        sample_dict = {}
        
        for i, (name, batch_dict) in enumerate(output_dict.items()):
            cur_decoder = self.motion_decoder_list[i]
            
            pred_ctrls, pred_scores = batch_dict['pred_list'][-1]
            
            best_idx = torch.argmax(pred_scores, dim=-1)
            
            # take value from the best index
            sample = pred_ctrls[torch.arange(pred_ctrls.shape[0]), best_idx, :3]
            
            sample = sample * cur_decoder.output_std + cur_decoder.output_mean
            
            sample_dict[name] = {'sample': sample}
        
        return sample_dict
    
    def to(self, device):
        """
        Move the model to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.context_encoder.to(device)
        for decoder in self.motion_decoder_list:
            decoder.to(device)
        return self
            
            
        