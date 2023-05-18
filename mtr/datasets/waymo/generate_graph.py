#!/usr/bin/env python
import os
import glob
import multiprocessing
from functools import partial
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import networkx as nx
from .pylanelet import PyLaneLet, PyLineString, PyLaneletMap
from .waymo_types import polyline_type
import warnings
from collections.abc import Iterable


def generate_boundary_line_string(map_infos: Dict) -> Tuple[Dict[int, PyLineString], Dict[int, PyLaneLet]]:
    '''
    This function generates dictionarys of boundary as line strings from the map info
    returns:
    
        boundary_line_string_dict: key is the feature feature_id, value is the PyLineString object
    '''
    boundary_line_string_dict = {}

    for road_line in map_infos['road_line']:
        feature_id = road_line['id']
        polyline = get_polyline(road_line, map_infos['all_polylines'])
        
        boundary_line_string_dict[feature_id] = PyLineString(
            feature_id, polyline, 'road_line', polyline_type[road_line['type']]
        )

    for road_edge in map_infos['road_edge']:
        feature_id = road_line['id']
        polyline = get_polyline(road_line, map_infos['all_polylines'])
        
        boundary_line_string_dict[feature_id] = PyLineString(
            feature_id, polyline, 'road_edge', polyline_type[road_edge['type']]
        )

    return boundary_line_string_dict


def parse_boundary(
    centerline: PyLineString,
    polyline: np.ndarray,
    boundaries: List,
    boundary_line_string_dict: Dict[int, PyLineString],
    boundary_info: Union[np.ndarray, None] = None,
    default_width: float = 0,
):
    '''
    This function parse the boundary information for each points on the centerline
    return: 
        boundary_info: Nx2 array, N is the number of points in the centerline
            the first column is the distance to the boundary (0 if boundary is unset)
            the second column is the boundary type (0 if boundary is unset)
    '''

    def get_closest_index(point):
        dis = np.linalg.norm(centerline.points[:, :3] - point, axis=-1)
        return np.argmin(dis)

    num_points = centerline.points.shape[0]
    # Initialize the boundary
    if boundary_info is None:
        boundary_info = np.zeros((num_points, 2))  # Default value is 0
        boundary_info[:, 0] = default_width

    for section in boundaries:
        # !! This is causing issues when we resample the centerline
        # get the start and end index of the centerline for this boundary section
        if polyline.shape[0] == num_points:
            start_feature_idx = section['start_index']
            end_feature_idx = section['end_index'] + 1
        else:
            start_feature_idx = get_closest_index(polyline[section['start_index'], :3])
            end_feature_idx = get_closest_index(polyline[section['end_index'], :3]) + 1
        centerline_points = centerline.points[start_feature_idx:end_feature_idx, :3]

        # retrieve the boundary line string

        if section['feature_id'] not in boundary_line_string_dict:
            # incase the feature_id is 0: boundary is not clearly defined
            continue

        boundary_line_string = boundary_line_string_dict[section['feature_id']]

        # calculate the distance from each point to the boundary
        try:
            boundary_points = boundary_line_string.distance_to_point(centerline_points)
        except ValueError:
            raise ValueError

        if np.max(boundary_points) > 5:
            warnings.warn("Boundary value are too large, clipped to 5. please check the boundary line string")
            boundary_points = np.clip(boundary_points, -np.inf, 5)

        # update the boundary info
        boundary_info[start_feature_idx:end_feature_idx, 0] = boundary_points
        boundary_info[start_feature_idx:end_feature_idx, 1] = boundary_line_string.sub_type

        # inform the boundary line string that it is used for current lane
        boundary_line_string.update_related_ids(centerline.feature_id)

    return boundary_info


def add_boundary_to_lanelet(lane, lanelet, boundary_line_string_dict):
    '''
    This function search the boundary information for each points on the lanelet centerline
    '''
    # if lane['id'] != 258:
    #     return
    centerline = lanelet.centerline

    default_width = 0
    if lane['type'] == 'TYPE_FREEWAY':
        default_width = 1.8
    elif lane['type'] == 'TYPE_SURFACE_STREET':
        default_width = 1.8
    elif lane['type'] == 'TYPE_BIKE_LANE':
        default_width = 0.6

    # ! Note: some lane boundary is not correctly defined in neighbor information. Hopefully it is fixed in the actual "road boundary" information
    for neighbor in lane['left_neighbors']:
        lanelet.left_boundary = parse_boundary(
            centerline,
            lane['polyline'],
            neighbor['boundaries'],
            boundary_line_string_dict,
            boundary_info=lanelet.left_boundary,
            default_width=default_width,
        )

    for neighbor in lane['right_neighbors']:
        lanelet.right_boundary = parse_boundary(
            centerline,
            lane['polyline'],
            neighbor['boundaries'],
            boundary_line_string_dict,
            boundary_info=lanelet.right_boundary,
            default_width=default_width,
        )

    lanelet.left_boundary = parse_boundary(
        centerline,
        lane['polyline'],
        lane['left_boundary'],
        boundary_line_string_dict,
        default_width=default_width,
    )
    lanelet.right_boundary = parse_boundary(
        centerline,
        lane['polyline'],
        lane['right_boundary'],
        boundary_line_string_dict,
        default_width=default_width,
    )

    # HACK: if only one side of the boundary is defined, we assume the other side has the same width
    idx_undefined = np.where(lanelet.left_boundary[:, 1] == 0)[0]
    lanelet.left_boundary[idx_undefined, 0] = lanelet.right_boundary[idx_undefined, 0]

    idx_undefined = np.where(lanelet.right_boundary[:, 1] == 0)[0]
    lanelet.right_boundary[idx_undefined, 0] = lanelet.left_boundary[idx_undefined, 0]


def prune_graph(
    map_graph: PyLaneletMap,
    max_distance: float = 500,
    max_polyline: int = 512,
    append_stop_sign: bool = True,
    remove_disconnected=True,
) -> PyLaneletMap:
    '''
    This function prunes the lanelet graph by removing lanelets that are too far from agents 
        and make sure that the total number of polyline is less than max_polyline
    Input:
        map_graph: a PyLaneletMap object
        agent_centers: a Nx3 array of agent centers
    '''
    if max_distance < 0 and max_polyline < 0:
        # no pruning is needed
        return map_graph

    pruned_graph = PyLaneletMap()

    lanelet_list = list(map_graph.lanelets.values())
    sorted_lanelet_list = sorted(lanelet_list, key=lambda x: x.distance_to_nearest_agent)

    add_lanelet = 0
    for lanelet in sorted_lanelet_list:
        if lanelet.distance_to_nearest_agent > max_distance:
            break

        num_lanelet_to_add = 1 + len(lanelet.stop_sign) * append_stop_sign
        if (add_lanelet + num_lanelet_to_add) > max_polyline and max_polyline > 0:
            break
        add_lanelet += num_lanelet_to_add
        pruned_graph.add_lanelet(lanelet)

    pruned_graph.build_graph()

    if remove_disconnected:
        # only keep lanelets that are connected to the first lanelet in the sorted list (typically there is an agent on it)
        pruned_graph.remove_disconnected_lanelets(
            anchor_id=sorted_lanelet_list[0].feature_id, type_to_ignore=[
                polyline_type['TYPE_BIKE_LANE'],
                polyline_type['TYPE_CROSSWALK'],
                polyline_type['TYPE_SPEED_BUMP'],
            ]
        )

    return pruned_graph


def get_polyline(feature, polylines):
    start_idx, end_idx = feature['polyline_index']
    return polylines[start_idx:end_idx]

def generate_map_graph(
    infos: Dict,
    max_point: int = -1,
    max_distance: float = -1,
    max_polyline: int = -1,
    remove_disconnected: bool = True,
    append_stop_sign: bool = True,
) -> PyLaneletMap:
    '''
    Main function to generate the map graph
    Input:
        map_infos, a dictionary of map info's
    Output:
        map_graph, a PyLanelet graph object
    '''

    map_infos = infos['map_infos']
    object_traj = infos['track_infos']['trajs']
    if infos['sdc_track_index'] not in infos['tracks_to_predict']['track_index']:
        index_of_interest = infos['tracks_to_predict']['track_index'] + [infos['sdc_track_index']]
    else:
        index_of_interest = infos['tracks_to_predict']['track_index']
    current_time_index = infos['current_time_index']
    agent_position = object_traj[index_of_interest, current_time_index, :3]

    # construct all boundary as line string for later use
    boundary_line_string_dict = generate_boundary_line_string(map_infos)

    lanelet_map = PyLaneletMap()

    # 1. Add lanelet that is within the max_distance to the lanelet_map

    for lane in map_infos['lane']:
        feature_id = lane['id']
        polyline = get_polyline(lane, map_infos['all_polylines'])
        lane['polyline'] = polyline
        # create a centerline for pyline string
        centerline = PyLineString(feature_id, polyline, 'lane', polyline_type[lane['type']])
        distance_to_nearest_agent = np.min(centerline.distance_to_point(agent_position))

        if distance_to_nearest_agent > max_distance and max_distance > 0:
            continue

        # create a lanelet object
        speed_limit = lane['speed_limit_mph'] * 0.44704
        lanelet = PyLaneLet(feature_id, centerline, polyline_type[lane['type']], speed_limit=speed_limit)

        # calculate the minimum eculedian distance from any of agents to the centerline
        lanelet.distance_to_nearest_agent = distance_to_nearest_agent

        # add infomation about lanelet's predecessor and successor
        lanelet.add_predecessor(lane['entry_lanes'])
        lanelet.add_successor(lane['exit_lanes'])

        # add information about lanelet's neighbor

        def get_neighbor_list(neighbors):
            return [neighbor['feature_id'] for neighbor in neighbors]
            # !Note: in waymo dataset, lane merge and divide also considered as neighbor, but this leads to issue when finding route
            neighbor_list = []
            for neighbor in neighbors:
                add = False
                # ! Hack here: check if boundary feature id is 0
                for boundary in neighbor['boundaries']:
                    if boundary['feature_id'] != 0:
                        add = True
                if add:
                    neighbor_list.append(neighbor['feature_id'])
            return neighbor_list

        lanelet.add_left(get_neighbor_list(lane['left_neighbors']), changable=True)
        lanelet.add_right(get_neighbor_list(lane['right_neighbors']), changable=True)

        # TODO: add neighbor lanelet that are not legally changeable (e.g. solid line)

        lanelet_map.add_lanelet(lanelet)

    # 2. Add stop sign info to lanelet map
    for stop_sign in map_infos['stop_sign']:
        for lanelet_id in stop_sign['lane_ids']:
            if lanelet_id in lanelet_map.lanelets:
                stop_sign['polyline'] = get_polyline(stop_sign, map_infos['all_polylines'])
                lanelet_map.get_lanelet(lanelet_id).add_stop_sign(stop_sign)

    # 3. Add cross walk and speed bump info to lanelet map
    for crosswalk in map_infos['crosswalk']:
        centerline = PyLineString(
            feature_id,
            get_polyline(crosswalk, map_infos['all_polylines']),
            'crosswalk',
            polyline_type['TYPE_CROSSWALK'],
        )
        distance_to_nearest_agent = np.min(centerline.distance_to_point(agent_position))
        if distance_to_nearest_agent > max_distance and max_distance > 0:
            continue

        # create a lanelet object
        lanelet = PyLaneLet(feature_id, centerline, polyline_type['TYPE_CROSSWALK'])
        lanelet.distance_to_nearest_agent = distance_to_nearest_agent

        lanelet_map.add_lanelet(lanelet)

    # ! Ignore speed bump for now
    # for feature_id, speed_bump in map_infos['speed_bump'].items():
    #     centerline = PyLineString(feature_id, speed_bump['polyline'], 'speed_bump', polyline_type['TYPE_SPEED_BUMP'])
    #     distance_to_nearest_agent = np.min(centerline.distance_to_point(agent_position))
    #     if distance_to_nearest_agent > max_distance and max_distance > 0:
    #         continue

    #     # create a lanelet object
    #     lanelet = PyLaneLet(feature_id, centerline, polyline_type['TYPE_SPEED_BUMP'])
    #     lanelet.distance_to_nearest_agent = distance_to_nearest_agent

    #     lanelet_map.add_lanelet(lanelet)

    # 4. build and prune the graph
    lanelet_map.build_graph()
    lanelet_map = prune_graph(
        lanelet_map,
        max_distance=max_distance,
        max_polyline=max_polyline,
        append_stop_sign=append_stop_sign,
        remove_disconnected=remove_disconnected,
    )

    # 5. Resample long polyline and add boundary information
    for lane in map_infos['lane']:
        feature_id = lane['id']
        if feature_id in lanelet_map.lanelets:
            lanelet = lanelet_map.get_lanelet(feature_id)

            # check if the centerline need to be resampled
            num_points = lanelet.centerline.points.shape[-2]
            num_stop_sign = len(lanelet.stop_sign)
            num_points_allow = max_point - num_stop_sign*append_stop_sign
            if max_point > 0 and num_points > num_points_allow:
                lanelet.centerline.resample(num_points_allow)

            # add boundary information
            add_boundary_to_lanelet(lane, lanelet, boundary_line_string_dict)

    return lanelet_map


def load_processed_scenario(scenario_file) -> Dict:
    '''
    This function is used to load the processed scenario file and return a dict
    '''
    with open(scenario_file, 'rb') as f:
        scenario = pickle.load(f)
    return scenario


