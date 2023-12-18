import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# set tf to cpu only
tf.config.set_visible_devices([], 'GPU')
import jax
jax.config.update('jax_platform_name', 'cpu')

import pickle
import numpy as np

# set jax device as cpu
from jax import numpy as jnp
from tqdm import tqdm
from typing import Dict, Tuple, List
from waymax import datatypes
from waymax import dataloader
from waymax import config as waymax_config
# from waymax.dynamics import bicycle_model
from rl_env.env_utils import compute_inverse
from matplotlib import pyplot as plt

def preprocess(
    serialized: bytes
) -> dict[str, tf.Tensor]:
    womd_features = dataloader.womd_utils.get_features_description(
        include_sdc_paths=False,
        max_num_rg_points=30000,
        num_paths=None,
        num_points_per_path=None,
    )
    womd_features['scenario/id'] = tf.io.FixedLenFeature([1], tf.string)

    deserialized = tf.io.parse_example(serialized, womd_features)
    parsed_id = deserialized.pop('scenario/id')
    deserialized['scenario/id'] = tf.io.decode_raw(parsed_id, tf.uint8)
    return dataloader.preprocess_womd_example(
        deserialized,
        aggregate_timesteps=True,
        max_num_objects=32,
    )

@jax.jit
def postprocess(example: dict[str, tf.Tensor]):
    scenario = dataloader.simulator_state_from_womd_dict(example)
    scenario_id = example['scenario/id']
    return scenario_id, scenario

@jax.jit
def find_all_valid(scenario: datatypes.SimulatorState):
    """
    Finds all valid objects that are vehicles and are moving in the given scenario.

    Parameters:
    - scenario: datatypes.SimulatorState
        The simulator state containing object metadata and trajectory information.

    Returns:
    - is_valid_vehicle_moving: bool
        A boolean array indicating whether each object is valid, a vehicle, and moving.
    """
    
    # is_sdc = scenario.object_metadata.is_sdc
    # is_modeled = scenario.object_metadata.is_modeled
    is_valid = scenario.object_metadata.is_valid
    is_vehicle = scenario.object_metadata.object_types == 1
    
    v_x_log = scenario.log_trajectory.vel_x # [num_agents, num_steps]
    v_y_log = scenario.log_trajectory.vel_y # [num_agents, num_steps]
    valid_log = scenario.log_trajectory.valid
    v_x_log = jnp.where(valid_log, v_x_log, 0)
    v_y_log = jnp.where(valid_log, v_y_log, 0)
    v_log = jnp.linalg.norm(jnp.stack([v_x_log, v_y_log], axis=-1), axis=-1)
    v_max = jnp.max(v_log, axis=-1)
    is_moving = v_max > 0.1
    
    return is_valid & is_vehicle & is_moving

@jax.jit
def find_pred(scenario: datatypes.SimulatorState):
    """
    Finds the predicted scenario based on the given SimulatorState.

    Parameters:
    - scenario: datatypes.SimulatorState
        The simulator state containing the scenario information.

    Returns:
    - bool
        True if the scenario is predicted, False otherwise.
    """
    is_sdc = scenario.object_metadata.is_sdc
    is_modeled = scenario.object_metadata.is_modeled
    is_vehicle = scenario.object_metadata.object_types == 1
    
    return is_sdc & is_modeled & is_vehicle

@jax.jit
def find_grid(scenario, t, interest_agent, accel_grid, steer_grid):
    """
    Finds the grid indices for the given scenario, time step, and validity conditions.

    Args:
        scenario: The scenario object.
        t: The time step.
        interest_agent: (num_agent,) A boolean array indicating the validity of each action.
        accel_grid: (num_accel_bins,) A 1D array containing the acceleration grid values.
        steer_grid: (num_steer_bins,) A 1D array containing the steering grid values.

    Returns:
        (num_agent, 3), A 2D array containing the grid indices and validity information.
    """
    
    action = compute_inverse(scenario.log_trajectory, t, 0.1, False)
    
    action_valid = action.valid.reshape(-1) & \
        (jnp.abs(action.data[:, 0]) < 15) & \
        (jnp.abs(action.data[:, 1]) < 0.45) & \
        interest_agent
    
    accel_idx = jnp.searchsorted(accel_grid, action.data[:, 0])
    steer_idx = jnp.searchsorted(steer_grid, action.data[:, 1])
    
    return jnp.stack([accel_idx, steer_idx, action_valid], axis=-1), action.data

def record_cache(key, dest, grid_idx, scenario_idx)->None:
    """
    Record cache information for a given key.

    Parameters:
        key (tuple): The key to identify the cache.
        dest (list): The list to store the cache information.
        grid_idx (ndarray): The grid index array.
        scenario_idx (int): The scenario index.
    """
    
    i, j = key
    t_idx_array, a_idx_array = np.where(
        (grid_idx[:, :, 0] == i) &
        (grid_idx[:, :, 1] == j) &
        (grid_idx[:, :, 2] == 1)
    )
    scenario_idx_array = np.ones_like(t_idx_array, dtype=int)*scenario_idx
    match_idx = np.stack([scenario_idx_array, t_idx_array, a_idx_array], axis=-1)
    dest.extend(match_idx.tolist())
    
def cache_one_scenario(
    scenario: datatypes.SimulatorState,
    scenario_idx: int,
    accel_grid: jax.Array,
    steer_grid: jax.Array,
    idx_cache: Dict[Tuple, List],
)->None:
    
    interest_agent = find_pred(scenario)
    
    # Find corresponding grid indices of each agnet at each time step of the scenario
    grid_idx, action_gt = jax.vmap(find_grid, in_axes=(None, 0, None, None, None))(scenario, jnp.arange(90), interest_agent, accel_grid, steer_grid)
    grid_idx = np.asarray(grid_idx)
    for key, dest in idx_cache.items():
        record_cache(key, dest, grid_idx, scenario_idx)
    return np.asarray(action_gt) # [T, num_agents, 2]
            
def cache_all_files(
    base_path: str,
    accel_grid: jax.Array,
    steer_grid: jax.Array,
    output_path: str,
):
    tf_dataset = dataloader.tf_examples_dataset(
        path=base_path,
        data_format=waymax_config.DataFormat.TFRECORD,
        preprocess_fn=preprocess,
        repeat=1,
        num_shards=8,
        deterministic=True,
    )
    os.makedirs(output_path, exist_ok=True)
        
    # initialize cache
    scenario_id_list = []
    idx_cache = {}
    for i in range(accel_grid.shape[0]):
        for j in range(steer_grid.shape[0]):
            idx_cache[(i, j)] = []
            
    for scenario_idx, example in enumerate(tqdm(tf_dataset.as_numpy_iterator())):
        scenario_id_binary, scenario = postprocess(example)
        scenario_id = scenario_id_binary.tobytes().decode('utf-8')
        scenario_id_list.append(scenario_id)
        action_gt = cache_one_scenario(scenario, scenario_idx, accel_grid, steer_grid, idx_cache)
        
        scenario_filename = os.path.join(output_path, 'scenario_'+scenario_id+'.pkl')
        with open(scenario_filename, 'wb') as f:
            pickle.dump({'scenario': scenario, 'action_gt': action_gt}, f)
            
    print('Extracted {} scenarios'.format(scenario_idx))
    
    # save cache
    output_dict = {
        'scenario_id_list': scenario_id_list,
        'idx_cache': idx_cache,
        'accel_grid': np.asarray(accel_grid),
        'steer_grid': np.asarray(steer_grid),
    }
    cache_filename = os.path.join(output_path, 'cache.pkl')
    with open(cache_filename, 'wb') as f:
        pickle.dump(output_dict, f)
        
    # plot histogram
    histogram = np.zeros((len(accel_grid), len(steer_grid)), dtype=int)
    for key, dest in idx_cache.items():
        i,j = key
        histogram[i,j] = len(dest)
        
    plt.imshow(np.log(histogram+1), extent=[accel_grid[0], accel_grid[-1], steer_grid[0], steer_grid[-1]], origin='lower', aspect='auto')
    plt.xlabel('acceleration')
    plt.ylabel('steering')
    plt.colorbar()
    fig_filename = os.path.join(output_path,'histogram.png')
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    accel_grid = jnp.linspace(-10, 10, 20)
    steer_grid = jnp.linspace(-0.3, 0.3, 20)
    
    # Training set
    base_path = '/Data/Dataset/Waymo/V1_2_tf/training/training_tfexample.tfrecord@1000'
    output_path = '/Data/Dataset/Waymo/V1_2_tf/training_extracted/'
    
    # Validation set
    # base_path = '/Data/Dataset/Waymo/V1_2_tf/validation/validation_tfexample.tfrecord@150'
    # output_path = '/Data/Dataset/Waymo/V1_2_tf/validation_extracted/'
    
    # base_path='/Data/Dataset/Waymo/V1_2_tf/validation_interactive/validation_interactive_tfexample.tfrecord@150',
    # output_path = '/Data/Dataset/Waymo/V1_2_tf/validation_interactive_extracted/'
    cache_all_files(base_path, accel_grid, steer_grid, output_path)
    