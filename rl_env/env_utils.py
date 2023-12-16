import numpy as np
import torch
from waymax import datatypes

import numpy as np
import jax
from typing import Dict, List
    
def merge_dict(batch_list: List, device: torch.device = 'cpu') -> Dict[str, torch.Tensor]:
    """Collects a batch of data from a list of transitions.

    Args:
        batch_list (List): a list of transitions.
        device (torch.device): device to store the data.

    Returns:
        Dict[str, torch.Tensor]: a batch of data.
    """
    list_len = len(batch_list)
    key_to_list = {}
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[i][key] for i in range(list_len)]

    input_batch = {}
    for key, value in key_to_list.items():
        val_type = type(value[0])
        if val_type == dict:
            input_batch[key] = merge_dict(value, device)
        elif val_type == torch.Tensor or val_type == np.ndarray or val_type == jax.Array:
            input_batch[key] = merge_batch_by_padding_2nd_dim(value).to(device)
        elif val_type == list:
            # stack list of lists
            input_batch[key] = [item for sublist in value for item in sublist]
        else:
            input_batch[key] = value
    return input_batch

def process_input(
    scenario: datatypes.SimulatorState,
    is_controlled: np.ndarray,
    from_gt: bool = False,
    current_time_index: int = None,
    visible_history: int = 1,
    history_length: int = 11,
    dt: float = 0.1,
) -> Dict:
    """
    Extracts scene data from a simulator state.

    Args:
        scenario (datatypes.SimulatorState): The simulator state.
        is_controlled (np.ndarray): A boolean array indicating which objects are controlled.
        history_length (int, optional): The number of timesteps to extract. Defaults to 11.
        dt (float, optional): The timestep. Defaults to 0.1.
        
    Returns:
        Dict: A dictionary containing the extracted scene data.
    """
    # Extract Objects Meta Data
    obj_metadata: datatypes.ObjectMetadata = scenario.object_metadata
    obj_types = np.asarray(obj_metadata.object_types)
    sdc_track_index = np.where(obj_metadata.is_sdc)[0][0] # only one sdc
    track_index_to_predict = np.where(is_controlled)[0]
    

    if from_gt:
        trajectory: datatypes.Trajectory = scenario.log_trajectory
        assert current_time_index is not None, "current_time_index must be provided when using ground truth"
    else:
        current_time_index = scenario.timestep
        # Extract Objects Trajectory
        trajectory: datatypes.Trajectory = scenario.sim_trajectory # TODO: check if this is the right trajectory
    
    timestamps = np.arange(history_length) * dt
    
    # Extract Objects State
    obj_trajs = _stack_traj(trajectory)
    
    end_index = current_time_index + 1
    start_index = max(0, end_index - history_length)
    obj_trajs_past = obj_trajs[:, start_index:end_index, :]
    center_objects = obj_trajs_past[track_index_to_predict, -1]
    
    if obj_trajs_past.shape[1] < history_length:
        # pad with zeros
        obj_trajs_past = np.pad(obj_trajs_past, ((0, 0), (history_length - obj_trajs_past.shape[1], 0), (0, 0))) 
    
    # create agent centric trajectory
    (obj_trajs_data, obj_trajs_mask, obj_trajs_last_pos) \
    = create_agent_data_for_center_objects(
        center_objects=center_objects,
        obj_trajs_past=obj_trajs_past, 
        track_index_to_predict=track_index_to_predict,
        sdc_track_index=sdc_track_index,
        timestamps=timestamps,
        obj_types=obj_types,
    )
    
    # Remove non seeing history
    # print(obj_trajs_data.shape, len(track_index_to_predict))
    # obj_trajs_data[np.arange(len(track_index_to_predict)), track_index_to_predict, :-visible_history, :] = 0.0
    # obj_trajs_mask[np.arange(len(track_index_to_predict)), track_index_to_predict, :-visible_history] = False
        
    polylines = _stack_map(scenario.roadgraph_points)
    
    # Extract the map information
    # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9),
    # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    map_polylines_data, map_polylines_mask, map_polylines_center = create_map_data_for_center_objects(
            center_objects=center_objects, 
            polylines=polylines,
            # center_offset=(0, 0), # !Hardcoded
        )   
    
    ret_dict = {
        'obj_trajs': obj_trajs_data,
        'obj_trajs_mask': obj_trajs_mask,
        'track_index_to_predict': track_index_to_predict,  # used to select center-features
        'obj_trajs_last_pos': obj_trajs_last_pos,
        'map_polylines': map_polylines_data,
        'map_polylines_mask': map_polylines_mask,
        'map_polylines_center': map_polylines_center,
    }
    
    return ret_dict
            
#################### Utils ####################
def _stack_traj(traj: datatypes.Trajectory) -> np.ndarray:
    """Stacks a trajectory into a 10D array."""
    # [cx, cy, cz, length, width, height, heading, vel_x, vel_y, valid]
    return np.stack([
        traj.x,
        traj.y,
        traj.z,
        traj.length,
        traj.width,
        traj.height,
        traj.yaw,
        traj.vel_x,
        traj.vel_y,
        traj.valid,
    ], axis=-1) # [num_tracks, num_steps, 10]
    
def _stack_map(map_data: datatypes.RoadgraphPoints) -> np.ndarray:
    return np.stack([
        map_data.x,
        map_data.y,
        map_data.z,
        map_data.dir_x,
        map_data.dir_y,
        map_data.dir_z,
        map_data.types,
        # map_data.ids,
        # map_data.valid,
    ], axis=-1) # [num_tracks, num_steps, 9]
    
def create_agent_data_for_center_objects(
        center_objects,
        obj_trajs_past,
        track_index_to_predict,
        sdc_track_index,
        timestamps,
        obj_types,    
    ):
    obj_trajs_data, obj_trajs_mask = generate_centered_trajs_for_agents(
        center_objects=center_objects, 
        obj_trajs_past=obj_trajs_past,
        obj_types=obj_types, 
        center_indices=track_index_to_predict,
        sdc_index=sdc_track_index,
        timestamps=timestamps, 
    )

    # generate the final valid position of each object
    obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
    num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
    obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
    for k in range(num_timestamps):
        cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
        obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

    return (
        obj_trajs_data,
        obj_trajs_mask > 0,
        obj_trajs_last_pos,
    )

def generate_centered_trajs_for_agents(
        center_objects,
        obj_trajs_past,
        obj_types,
        center_indices,
        sdc_index,
        timestamps
    ):
    """
    This function create 'agent-centric' trajectories for each object around each center objects
    Args:
        center_objects (num_center_objects, 10): the current states of objects that need to be predicted
            [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid], 
        obj_trajs_past (num_objects, num_timestamps, 10): the past trajectories of all objects
        obj_types: List[str] (num_objects): the type of each object
        center_indices (num_center_objects): the index of center objects in obj_trajs_past
        sdc_index (int): the index of the self-driving car in obj_trajs_past
        timestamps ([float]): list of timestamps for the past trajectories
        
    Returns:
        ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
    """
    # assert obj_trajs_past.shape[-1] == 10
    # assert center_objects.shape[-1] == 10
    num_center_objects = center_objects.shape[0]
    num_objects, num_timestamps, _ = obj_trajs_past.shape
    
    # transform the coordinate systems to each centered objects
    # [num_center_objects, num_objects, num_timestamps, num_attrs]
    obj_trajs = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_past,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )

    ## generate the attributes for each object
    object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
    object_onehot_mask[:, obj_types == 1, :, 0] = 1
    object_onehot_mask[:, obj_types == 2, :, 1] = 1  # TODO: CHECK THIS TYPO
    object_onehot_mask[:, obj_types == 3, :, 2] = 1
    object_onehot_mask[np.arange(num_center_objects), center_indices, :, 3] = 1
    if sdc_index is not None:
        object_onehot_mask[:, sdc_index, :, 4] = 1

    object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    object_time_embedding[:, :, np.arange(num_timestamps), np.arange(num_timestamps)] = 1
    object_time_embedding[:, :, np.arange(num_timestamps), -1] = timestamps

    object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
    vel_pre = np.roll(vel, shift=1, axis=2)
    acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
    acce[:, :, 0, :] = acce[:, :, 1, :]

    ret_obj_trajs = np.concatenate((
        obj_trajs[:, :, :, 0:6], 
        object_onehot_mask,
        object_time_embedding, 
        object_heading_embedding,
        obj_trajs[:, :, :, 7:9], 
        acce,
    ), axis=-1)

    ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  
    ret_obj_trajs[ret_obj_valid_mask == 0] = 0

    return ret_obj_trajs, ret_obj_valid_mask

def transform_trajs_to_center_coords(
    obj_trajs,
    center_xyz,
    center_heading,
    heading_index,
    rot_vel_index=None,
):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    return:
        obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
    """
    num_objects, num_timestamps, num_attrs = obj_trajs.shape
    num_center_objects = center_xyz.shape[0]
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    obj_trajs = np.copy(obj_trajs)[None].repeat(num_center_objects, axis=0)
    obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    obj_trajs[:, :, :, 0:2] = rotate_points_along_z(
        points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
        angle=-center_heading
    ).reshape(num_center_objects, num_objects, num_timestamps, 2)

    obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]
    
    obj_trajs[:, :, :, heading_index] = np.arctan2(
        np.sin(obj_trajs[:, :, :, heading_index]),
        np.cos(obj_trajs[:, :, :, heading_index])
    )

    # rotate direction of velocity
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2
        obj_trajs[:, :, :, rot_vel_index] = rotate_points_along_z(
            points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

    return obj_trajs

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = np.stack((
            cosa,  sina,
            -sina, cosa
        ), axis=1).reshape(-1, 2, 2)
        points_rot = np.matmul(points, rot_matrix)
    else:
        ones = np.ones(points.shape[0])
        rot_matrix = np.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), axis=1).reshape(-1, 3, 3)
        points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot

def generate_batch_polylines_from_map(
    polylines, point_sampled_interval=1,
    vector_break_dist_thresh=1.0, num_points_each_polyline=20
):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    return ret_polylines, ret_polylines_mask

def create_map_data_for_center_objects(center_objects, polylines):
    """
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        map_infos (dict):
            all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        center_offset (2):, [offset_x, offset_y]
    Returns:
        map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
    """
    num_center_objects = center_objects.shape[0]

    # transform object coordinates by center objects
    def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
        neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        neighboring_polylines[:, :, :, 0:2] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 0:2].reshape((num_center_objects, -1, 2)),
            angle=-center_objects[:, 6]
        ).reshape((num_center_objects, -1, batch_polylines.shape[1], 2))
        neighboring_polylines[:, :, :, 3:5] = rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].reshape((num_center_objects, -1, 2)),
            angle=-center_objects[:, 6]
        ).reshape((num_center_objects, -1, batch_polylines.shape[1], 2))
        
        # use pre points to map
        # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
        xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        neighboring_polylines = np.concatenate([neighboring_polylines, xy_pos_pre], axis=-1)
        neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
        return neighboring_polylines, neighboring_polyline_valid_mask

    # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)
    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        polylines=polylines,
    )  

    # collect a number of closest polylines for each center objects
    num_of_src_polylines = 768
    
    def get_polyline_center(map_polylines, map_polylines_mask):
        '''
        map_polylines: (..., num_points_each_polyline, 9)
        map_polylines_mask: (..., num_points_each_polyline)
        
        return: (..., 3)
        '''
        temp_sum = (map_polylines[..., 0:3] * map_polylines_mask[..., None]).sum(axis=-2) 
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1)[..., None], a_min=1.0, a_max=None) 
        return map_polylines_center

    if len(batch_polylines) > num_of_src_polylines:
        polyline_center = get_polyline_center(batch_polylines, batch_polylines_mask)
        pos_of_map_centers = center_objects[:, 0:2]  # (num_center_objects, 2)
        dist = np.linalg.norm(pos_of_map_centers[:, None, :] - polyline_center[None, :, :2], axis=-1)  # (num_center_objects, num_polylines)
        topk_idxs = np.argsort(dist, axis=-1)[:, :num_of_src_polylines]  # (num_center_objects, num_topk_polylines)
        map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    else:
        map_polylines = np.repeat(batch_polylines[None, ...], num_center_objects, axis=0)
        map_polylines_mask = np.repeat(batch_polylines_mask[None, ...], num_center_objects, axis=0)

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask
    )
 
    # (num_center_objects, num_polylines, 3)
    map_polylines_center = get_polyline_center(map_polylines, map_polylines_mask)
    
    return map_polylines, map_polylines_mask, map_polylines_center

def merge_batch_by_padding_2nd_dim(tensor_list):
    ret_tensor_list = []
    if len(tensor_list[0].shape) > 1:
        maxt_feat0 = max([x.shape[1] for x in tensor_list])
        rest_size = tensor_list[0].shape[2:]
        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k]
            if type(cur_tensor) == np.ndarray:
                cur_tensor = torch.from_numpy(cur_tensor.copy())
            elif type(cur_tensor) == jax.Array:
                cur_tensor = torch.from_numpy(np.asarray(cur_tensor).copy())
            assert cur_tensor.shape[2:] == rest_size

            new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, *rest_size)
            new_tensor[:, :cur_tensor.shape[1], ...] = cur_tensor
            ret_tensor_list.append(new_tensor)
    else:
        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k]
            if type(cur_tensor) == np.ndarray:
                cur_tensor = torch.from_numpy(cur_tensor)
            elif type(cur_tensor) == jax.Array:
                cur_tensor = torch.from_numpy(np.asarray(cur_tensor))
            ret_tensor_list.append(cur_tensor)      
    return torch.cat(ret_tensor_list, dim=0).contiguous()  

