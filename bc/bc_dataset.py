import torch
from torch.utils.data import IterableDataset
import pickle
import os
from rl_env.env_utils import merge_dict, process_input
from waymax import datatypes
import numpy as np
from typing import List
import jax

class BCDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        sample_method: str = 'log',
        count_cap: float = None
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
        self.histogram = np.clip(self.histogram, 0, count_cap)
        
        if sample_method == 'uniform':
            self.sample_p = self.histogram > 0
        elif sample_method == 'log':
            self.sample_p = np.log(self.histogram+1)
        else:
            self.sample_p = self.histogram
        # normalize
        self.sample_p = self.sample_p / self.sample_p.sum()
                    
    def retrive_one(self, cache: List, hide_history: int = 1, with_scenario: bool = False):
        """
        Retrieves a single example from the dataset based on the given cache.

        Args:
            cache (List): A list containing the indices used to retrieve the example.

        Returns:
            dict: A dictionary containing the retrieved example and additional information.
        """
        scenario_idx = cache[0]
        scenario_id = self.scenario_id_list[scenario_idx]
        
        a_idx = cache[1]
        t_idx = cache[2]
        
        scenario_filename = os.path.join(self.data_path, 'scenario_'+scenario_id+'.pkl')
        
        # load scenario
        with open(scenario_filename, 'rb') as f:
            scenario_dict = pickle.load(f)
        scenario: datatypes.SimulatorState = scenario_dict['scenario']
        
        full_action_gt: np.ndarray = scenario_dict['action_gt'] #(T, A, 2)
                
        is_controlled = np.zeros(32, dtype=bool)
        is_controlled[a_idx] = True 
        input_dict = process_input(
            scenario=scenario,
            is_controlled=is_controlled,
            from_gt=True,
            current_time_index=t_idx,
            hide_history=hide_history,
        )
        
        input_dict['gt_action'] = full_action_gt[a_idx, t_idx:t_idx+1, :]
        
        input_dict['scenario_id'] = [scenario_id]
        input_dict['t'] = [t_idx]
        
        if with_scenario:
            return input_dict, scenario
        else:
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


class BCDatasetTester(BCDataset):
    def __init__(
        self,
        data_path: str,
        count_cap: float = 0.1
    ) -> None:
        super().__init__(data_path, count_cap)
        
    def __iter__(self):
        while True:

            bin_idx = np.random.choice(len(self.sample_p), p=self.sample_p)
            cache_id = np.random.randint(self.histogram[bin_idx])
            cache = self.idx_cache[bin_idx][cache_id]
            scenario_idx = cache[0]
            a_idx = cache[1]
            t_idx = cache[2]
            scenario_id = self.scenario_id_list[scenario_idx]
            
            scenario_filename = os.path.join(self.data_path, 'scenario_'+scenario_id+'.pkl')
            
            # load scenario
            with open(scenario_filename, 'rb') as f:
                scenario_dict = pickle.load(f)
            full_action_gt: np.ndarray = scenario_dict['action_gt'] #(T, A, 2)
            yield bin_idx, cache, full_action_gt[a_idx, t_idx, :]
            
    def collate_fn(self, batch_list):
        bin_list = []
        cache_list = []
        action_list = []
        for bin_idx, cache, action in batch_list:
            bin_list.append(bin_idx)
            cache_list.append(cache)
            action_list.append(action)
        return bin_list, cache_list, np.stack(action_list, axis=0)
        