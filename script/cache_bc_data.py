# Import Network
import jax
import os
from jax import numpy as jnp
import numpy as np
import dataclasses
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics
from waymax import dataloader
from waymax import visualization
import torch
import pickle
from waymax.dynamics import bicycle_model
from rl_env.env_utils import process_input, create_iter
from tqdm import tqdm


def cache_data_set(tf_iter, path):
    os.makedirs(path, exist_ok=True)
    count = 0
    for scenario_id, scenario in (pbar := tqdm(tf_iter)):
        pbar.set_description(f"Processing {scenario_id}")
        scenario: datatypes.SimulatorState
        
        for cur_t in range(90):
            # Get GT action by inverse kinematics
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
            
                filename = f'{scenario_id}_{cur_t:02d}_{idx:02d}.pkl'
                
                with open(os.path.join(path, filename), 'ab') as f:
                    pickle.dump(cur_input_dict, f)
            count += valid_agent.sum()
            
    print(f"total {count} data generated")

if __name__ == "__main__":
    cache_path = '/Data/Dataset/BC/train'
    
    WOMD_1_2_0_TRAIN_LOCAL = _config.DatasetConfig(
        path='/Data/Dataset/Waymo/V1_2_tf/training/training_tfexample.tfrecord@1000',
        max_num_rg_points=30000,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=32,
        repeat= 1,
    )
    tf_iter = create_iter(WOMD_1_2_0_TRAIN_LOCAL)
    cache_data_set(tf_iter, cache_path)
    