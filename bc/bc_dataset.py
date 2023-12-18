import torch
from torch.utils.data import IterableDataset
import pickle
import os
from rl_env.env_utils import merge_dict, process_input
from waymax import datatypes
import numpy as np
from typing import List
import jax

from waymax.dynamics.bicycle_model import compute_inverse



class BCDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        sample_method: str = 'uniform'
    ):
        # Get all files in the directory
        self.data_path = data_path
        
        # load cache file
        cache_file = os.path.join(data_path, 'cache.pkl')
        with open(cache_file, 'rb') as f:
            self.cache = pickle.load(f)
        self.scenario_id_list = self.cache['scenario_id_list']
        self.accel_grid = self.cache['accel_grid']
        self.steer_grid = self.cache['steer_grid']
        
        # create_histogram
        self.histogram = []
        self.idx_cache = {}
        for i, idx_list in enumerate(self.cache['idx_cache'].values()):
            self.histogram.append(len(idx_list))
            self.idx_cache[i] = idx_list
        self.histogram = np.asarray(self.histogram)
        
        if sample_method == 'uniform':
            self.sample_p = self.histogram > 0
        elif sample_method == 'log':
            self.sample_p = np.log(self.histogram+1)
        else:
            self.sample_p = self.histogram
        # normalize
        self.sample_p = self.sample_p / self.sample_p.sum()
        
        self.compute_inverse_jit = jax.jit(compute_inverse, static_argnums=(3))
            
    def retrive_one(self, cache: List, sanity_check: bool = False):
        """
        Retrieves a single example from the dataset based on the given cache.

        Args:
            cache (List): A list containing the indices used to retrieve the example.

        Returns:
            dict: A dictionary containing the retrieved example and additional information.
        """
        scenario_idx = cache[0]
        scenario_id = self.scenario_id_list[scenario_idx]
        t_idx = cache[1]
        a_idx = cache[2]
        
        scenario_filename = os.path.join(self.data_path, 'scenario_'+scenario_id+'.pkl')
        
        # load scenario
        with open(scenario_filename, 'rb') as f:
            scenario_dict = pickle.load(f)
        scenario: datatypes.SimulatorState = scenario_dict['scenario']
        # full_action_gt: np.ndarray = scenario_dict['action_gt'] #(T, A, 2)
                
        is_controlled = np.zeros(32, dtype=bool)
        is_controlled[a_idx] = True 
        input_dict = process_input(
            scenario=scenario,
            is_controlled=is_controlled,
            from_gt=True,
            current_time_index=t_idx,
        )
        
        # input_dict['gt_action'] = full_action_gt[t_idx, a_idx:a_idx+1, :]
        input_dict['gt_action'] = np.asarray(self.compute_inverse_jit(scenario.log_trajectory, t_idx, 0.1, False).data)[a_idx:a_idx+1]
        
        # Verify the control
        if sanity_check:
            action_ref = compute_inverse(scenario.log_trajectory, t_idx).data[a_idx]
            assert np.allclose(action_ref, input_dict['gt_action'], atol=1e-2), f'in dataset, action_ref: {action_ref}, gt_action: {input_dict["gt_action"]}'
        
        input_dict['scenario_id'] = [scenario_id]
        input_dict['t'] = [t_idx]
        
        return input_dict
        
    def collate_fn(self, batch_list):
        input_dict = merge_dict(batch_list)
        return input_dict
    
    def __iter__(self):
        while True:
            bin_idx = np.random.choice(len(self.sample_p), p=self.sample_p)
            cache_id = np.random.randint(self.histogram[bin_idx])
            cache = self.idx_cache[bin_idx][cache_id]
            input_dict = self.retrive_one(cache)
            yield input_dict

