import torch
from torch.utils.data import Dataset, IterableDataset
import pickle
import glob
import os
from rl_env.env_utils import merge_dict
from rl_env.env_utils import process_input, create_iter
from collections import deque 
from waymax import datatypes
from waymax.dynamics import bicycle_model

import numpy as np
from jax import numpy as jnp

class BCDataset(Dataset):
    def __init__(self, data_path):
        # Get all files in the directory
        self.files = glob.glob(os.path.join(data_path, '*.pkl'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)
        return data
    
    def collate_fn(self, batch_list):
        input_dict = merge_dict(batch_list)
        return input_dict
        
class BCDatasetBuffer(IterableDataset):
    def __init__(self, data_config, buffer_size=1000):
        self.data_config = data_config
        self.data_iter = create_iter(data_config)
        self.buffer_size = buffer_size
        self.out_buffer = []
        self.in_buffer = deque(maxlen=None)
        
    def extract_data(self):
        # Get GT action by inverse kinematics
        scenario_id, scenario = next(self.data_iter)
        for cur_t in range(90):
            action = bicycle_model.compute_inverse(scenario.log_trajectory, cur_t)
            valid_agent = jnp.logical_and(action.valid.reshape(-1), scenario.object_metadata.object_types)
            gt_action = np.asarray(action.data)
            if not valid_agent.any():
                return None

            input_dict = process_input(
                scenario=scenario,
                is_controlled=valid_agent,
                from_gt=True,
                current_time_index=cur_t,
            )
            
            # Split data to each agent
            for i, idx in enumerate(jnp.where(valid_agent)[0]):
                cur_input_dict = {k: v[i:i+1] for k, v in input_dict.items()}
                cur_input_dict['gt_action'] = gt_action[idx:idx+1]
                cur_input_dict['scenario_id'] = [scenario_id]
                cur_input_dict['t'] = [cur_t]
                
                self.in_buffer.append(cur_input_dict)
    
    def get_from_in_buffer(self):
        if len(self.in_buffer) == 0:
            self.extract_data()
        return self.in_buffer.popleft()
        
    def __iter__(self):
        while True:
            while len(self.out_buffer) < self.buffer_size:
                self.out_buffer.append(self.get_from_in_buffer())
                
            idx = np.random.randint(self.buffer_size)
            item = self.out_buffer[idx]
            yield item
            self.out_buffer[idx] = self.get_from_in_buffer()
                
    def collate_fn(self, batch_list):
        input_dict = merge_dict(batch_list)
        return input_dict
        