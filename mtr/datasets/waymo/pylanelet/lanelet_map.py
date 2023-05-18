import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from .lanelet import PyLaneLet
from functools import lru_cache
import random
from pyspline import Curve
from typing import List


class PyLaneletMap:

    def __init__(self) -> None:
        self.lanelets = {}
        self.routing_graph = nx.DiGraph()
        self.graph_initialized = False
        self.psi_weight = 2  # weight for heading difference in cost function

    def update_lanelet(self, lanelet_dict):
        '''
        Update lanelet information
        '''
        self.lanelets.update(lanelet_dict)

    def add_lanelet(self, lanelet: PyLaneLet):
        self.lanelets[lanelet.feature_id] = lanelet

    def get_lanelet(self, lanelet_id: int) -> PyLaneLet:
        return self.lanelets[lanelet_id]

    def get_closest_lanelet(self, pose, check_psi: bool = False) -> PyLaneLet:
        '''
        Get the closest lanelet to the given pose
        Parameters:
            pose: array, [x, y, z, (optional) psi]
            check_psi: bool, whether to check heading difference
        Returns:
            closest_lanelet: PyLaneLet, closest lanelet
            lanelet_s: float, normalized position on the centerline
            signed_distance: float, signed distance to the centerline
            
        '''
        pose = np.array(pose, dtype=np.float32)
        min_dist = float('inf')
        closest_lanelet = None
        lanelet_s = None
        signed_distance = None
        point = pose[:-1]
        if check_psi:
            psi = pose[-1]

        for lanelet in self.lanelets.values():
            if lanelet.centerline.feature_type != 'lane':
                continue
            
            spline = lanelet.centerline.spline
            
            if spline is None:
                continue
            s, d = spline.projectPoint(point)
            dist = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            deriv = spline.getDerivative(s)
            ref_psi = np.arctan2(deriv[1], deriv[0])

            # check heading
            if check_psi:

                psi_dist = psi - ref_psi
                psi_dist = self.psi_weight * np.abs(np.arctan2(np.sin(psi_dist), np.cos(psi_dist)))
            else:
                psi_dist = 0

            total_dist = dist + psi_dist

            if total_dist < min_dist:
                min_dist = total_dist
                closest_lanelet = lanelet
                lanelet_s = s
                signed_distance = -np.sin(ref_psi) * d[0] + np.cos(ref_psi) * d[1]

        return closest_lanelet, lanelet_s, signed_distance

    def build_graph(self, lane_change_cost=3):
        '''
        Build networkx graph for routing
        '''
        # print("Building routing graph with lane change cost: ", lane_change_cost)
        lanelet: PyLaneLet
        for lanelet in self.lanelets.values():
            cur_id = lanelet.feature_id
            self.routing_graph.add_node(cur_id)
            # lane change
            for left_id in lanelet.left_changable:
                if left_id in self.lanelets:
                    self.routing_graph.add_edge(cur_id, left_id, length=np.inf, lane_change = lane_change_cost)
                # else:
                #     print("Left lanelet {} not found".format(left_id))
            for right_id in lanelet.right_changable:
                if right_id in self.lanelets:
                    self.routing_graph.add_edge(cur_id, right_id, length=np.inf, lane_change = lane_change_cost)
                # else:
                #     print("Right lanelet {} not found".format(right_id))
            # lane follow
            for successor_id in lanelet.successor:
                if successor_id in self.lanelets:
                    self.routing_graph.add_edge(cur_id, successor_id, length=lanelet.length, lane_change = lanelet.length)

            for predecessor_id in lanelet.predecessor:
                if predecessor_id in self.lanelets:
                    self.routing_graph.add_edge(predecessor_id, cur_id, length=lanelet.length, lane_change = lanelet.length)
                # else:
                #     print("Successor lanelet {} not found".format(successor_id))

        self.graph_initialized = True

    def remove_disconnected_lanelets(self, anchor_id: int = None, type_to_ignore: List = []):
        '''
        Remove lanelets that are not connected to the rest of the map
        input:
            anchor_id: int, id of the lanelet whose connected components will be kept
            type_to_ignore: List, list of lanelet types to keep even if they are not connected
        '''
        if not self.graph_initialized:
            self.build_graph()

        # extract subgraphs
        sub_graphs = nx.connected_components(nx.Graph(self.routing_graph))
        anchor_id = None if anchor_id not in self.routing_graph else anchor_id

        if anchor_id is None:
            for i, c in enumerate(sorted(sub_graphs, key=len, reverse=True)):
                if i == 0:
                    self.routing_graph = self.routing_graph.subgraph(c).copy()
                else:
                    for id in c:  # remove lanelets that are not in the largest subgraph
                        if self.lanelets[id].type not in type_to_ignore:
                            self.lanelets.pop(id, None)
        else:
            for c in sub_graphs:
                if anchor_id in c:
                    self.routing_graph = self.routing_graph.subgraph(c).copy()
                else:
                    for id in c:
                        if self.lanelets[id].type not in type_to_ignore:
                            self.lanelets.pop(id, None)

    def get_shortest_route(self, start_id: int, end_id: int):
        '''
        Get the shortest route from start_id to end_id
        '''
        def _get_shortest_route(metric):
            route = nx.shortest_path(self.routing_graph, start_id, end_id, weight=metric)
            route_length = 0
            for i in range(len(route) - 1):
                route_length += self.routing_graph[route[i]][route[i + 1]][metric]
            return route, route_length
        
        route, route_length = _get_shortest_route('length')
        if route_length == np.inf:
            route, route_length = _get_shortest_route('lane_change')

        return route, route_length

    def get_shortest_path(self, start, end, start_pose: bool = False, end_pose: bool = False, verbose: bool = True):
        '''
        Get the shortest path from start to end
        '''
        if not self.graph_initialized:
            self.build_graph()

        centerline_list = []
        start_lanelet, start_s, _ = self.get_closest_lanelet(start, check_psi=start_pose)
        end_lanelet, end_s, _ = self.get_closest_lanelet(end, check_psi=end_pose)

        route = None

        # check if start and end are in the same lanelet or they are neighbors
        if (start_lanelet.feature_id == end_lanelet.feature_id \
                or end_lanelet.feature_id in start_lanelet.left \
                or end_lanelet.feature_id in start_lanelet.right):
            # if abs(start_s - end_s)*start_lanelet.length < 0.1:
            #     if verbose:
            #         print("Start and end points are too close")
            #     return None
            if end_s <= start_s:
                centerline_list.append(self.get_reference(start_lanelet, start_s, 1, endpoint=False))
                route_cost = float('inf')
                # search through successors for the best route
                for successor_id in start_lanelet.successor:
                    temp_route, temp_route_cost = self.get_shortest_route(successor_id, end_lanelet.feature_id)
                    if temp_route_cost < route_cost:
                        route = temp_route
                        route_cost = temp_route_cost
                # new start point is the beginning of the best successor
                start_s = 0
        # Get the shortest route if we have not found one yet
        if route is None:
            route, _ = self.get_shortest_route(start_lanelet.feature_id, end_lanelet.feature_id)

        num_lanelets = len(route)
        for i in range(num_lanelets - 1):
            cur_lanelet = self.lanelets[route[i]]
            next_lanelet_id = route[i + 1]
            # check if we need to change lane
            if next_lanelet_id in cur_lanelet.left or next_lanelet_id in cur_lanelet.right:
                if i == num_lanelets - 2:
                    # change lane to the last lanelet
                    cur_end = (end_s-start_s) * 0.3 + start_s
                    next_start = (end_s-start_s) * 0.7 + start_s
                else:
                    cur_end = (1-start_s) * 0.3 + start_s
                    next_start = (1-start_s) * 0.7 + start_s
            else:
                cur_end = 1
                next_start = 0
            centerline_list.append(self.get_reference(cur_lanelet, start_s, cur_end, endpoint=False))
            start_s = next_start

        centerline_list.append(self.get_reference(end_lanelet, start_s, end_s, endpoint=True))

        centerline = np.concatenate(centerline_list, axis=0)
        # In case of repeated points, pyspline will throw an error
        _, idx = np.unique(np.round(centerline[:, :2], 3), axis=0, return_index=True)
        centerline_unique = centerline[np.sort(idx), :]
        return centerline_unique

    def get_route_centerline(self, route, start_s, end_s=1):
        '''
        Given a route, return the centerline of the route
        
        '''
        centerline_list = []
        num_lanelets = len(route)
        for i in range(num_lanelets - 1):
            cur_lanelet = self.lanelets[route[i]]
            next_lanelet_id = route[i + 1]
            # check if we need to change lane
            if next_lanelet_id in cur_lanelet.left or next_lanelet_id in cur_lanelet.right:
                if i == num_lanelets - 2:
                    # change lane to the last lanelet
                    cur_end = (end_s-start_s) * 0.3 + start_s
                    next_start = (end_s-start_s) * 0.7 + start_s
                else:
                    cur_end = (1-start_s) * 0.3 + start_s
                    next_start = (1-start_s) * 0.7 + start_s
            else:
                cur_end = 1
                next_start = 0
            centerline_list.append(self.get_reference(cur_lanelet, start_s, cur_end, endpoint=False))
            start_s = next_start

        end_lanelet = self.lanelets[route[-1]]
        centerline_list.append(self.get_reference(end_lanelet, start_s, end_s, endpoint=True))

        centerline = np.concatenate(centerline_list, axis=0)
        # In case of repeated points, pyspline will throw an error
        _, idx = np.unique(np.round(centerline[:, :2], 3), axis=0, return_index=True)
        centerline_unique = centerline[np.sort(idx), :]
        return centerline_unique

    def get_reference(self, lanelet, start_s, end_s, endpoint: bool = False):
        '''
        Get the reference line of a lanelet 
        Parameters:
            lanlet: the lanelet
            start_s: the start s value
            end_s: the end s value
        return:
            reference line:[Nx5], [x,y,left_width,right_width,speed_limit]
        '''
        dl = 0.1
        num_pt = int(max(2, (end_s-start_s) * lanelet.length / dl))
        centerline = lanelet.get_section_centerline(start_s, end_s, endpoint=endpoint, num=num_pt)

        cur_width = lanelet.get_section_width(start_s, end_s, endpoint=endpoint, num=num_pt)
        left_width = cur_width / 2
        right_width = cur_width / 2

        for left_id in lanelet.left:
            left_lanelet = self.lanelets[left_id]
            left_width += left_lanelet.get_section_width(start_s, end_s, endpoint=endpoint, num=num_pt)

        for right_id in lanelet.right:
            right_lanelet = self.lanelets[right_id]
            right_width += right_lanelet.get_section_width(start_s, end_s, endpoint=endpoint, num=num_pt)

        left_width = left_width[:, np.newaxis]
        right_width = right_width[:, np.newaxis]
        speed_limit = lanelet.speed_limit * np.ones_like(left_width)

        reference = np.concatenate([centerline, left_width, right_width, speed_limit], axis=1)
        return reference

    def get_random_waypoint(self, with_neighbors: bool = False):
        '''
        Randomly select a (valid) waypoint on the map
        Return:
            waypoint: [x,y,psi]
        '''

        # Randomly select a lanelet
        if with_neighbors:
            has_neighbor = False
            while has_neighbor == False:
                lanelet_id = random.choice(list(self.lanelets.keys()))
                lanelet = self.lanelets[lanelet_id]
                if len(lanelet.left) > 0 or len(lanelet.right) > 0:
                    has_neighbor = True
                s = np.random.uniform(0.1, 0.9)
        else:
            lanelet_id = random.choice(list(self.lanelets.keys()))
            s = np.random.uniform(0, 1)
        terminal_lanelet = self.lanelets[lanelet_id]

        return terminal_lanelet.centerline.get_ref_pose(s)

    def get_reachable_path(self, start_id, start_s, d_max, allow_lane_change=False):
        '''
        Get the closest lanelet to the given pose
        Parameters:
            pose: array, [x, y, (optional) psi]
            d_max: the maximum distance to search
            allow_lane_change: whether to allow lane change
        Return:
            a list of pyspline Curve
        '''

        # apply the memoization
        @lru_cache(maxsize=None)
        def all_paths_to_target(cur_id, d_prev, lane_change):
            results = []
            # check lane change
            if lane_change and allow_lane_change:  # do this to avoid repeated lane change
                for left_id in self.lanelets[cur_id].left:
                    for path in all_paths_to_target(left_id, d_prev + 0.5, False):
                        results.append([cur_id] + path)
                for right_id in self.lanelets[cur_id].right:
                    for path in all_paths_to_target(right_id, d_prev + 0.5, False):
                        results.append([cur_id] + path)

            cur_lanelet = self.lanelets[cur_id]
            d_cur = d_prev + cur_lanelet.length
            if d_cur >= d_max:
                results.append([cur_id])
                return results

            for successor_id in self.lanelets[cur_id].successor:
                for path in all_paths_to_target(successor_id, d_cur, allow_lane_change):
                    results.append([cur_id] + path)

            return results

        # get the closest lanelet
        start_lanelet = self.lanelets[start_id]
        reachable_routes = all_paths_to_target(start_id, -1 * start_s * start_lanelet.length, allow_lane_change)

        # get the centerline of all reachable path
        centerline_list = []
        for route in reachable_routes:
            centerline = self.get_route_centerline(route, start_s)
            route_curve = Curve(x=centerline[:, 0], y=centerline[:, 1], k=3)
            centerline_list.append(route_curve)

        return centerline_list

    def generate_feature(self, max_polyline: int = -1, max_point: int = -1, append_stop_sign: bool = False):
        '''
        This function generates the feature and mask for the map
        Input: 
            max_polyline: int, maximum number of polylines in the map
            max_point: int, maximum number of points in the polyline feature
            append_stop_sign: bool, whether to append stop sign feature to the end of the polyline feature
        Output:
            feature: (max_polyline, max_point, 11 ) array, 
                polyline feature [x, y, z, dx, dy, dz, speed limit, width_L, width_R, lane_type, left_type, right_type, feature_id]
            mask: (max_polyline, max_point) array, 1 for valid point, 0 for invalid point
        '''
        feature_list = []
        mask_list = []

        for lanelet in self.lanelets.values():
            feature, mask = lanelet.generate_feature(max_point, append_stop_sign)
            feature_list.append(feature)
            mask_list.append(mask)

        # concatenate all features
        full_feature = np.concatenate(feature_list, axis=0)
        full_mask = np.concatenate(mask_list, axis=0)

        num_polyline = full_feature.shape[0]
        assert num_polyline <= max_polyline, "Number of polylines exceeds the limit"

        output_feature = np.zeros((max_polyline, max_point, 11), dtype=np.float32)
        output_feature[:num_polyline, :, :] = full_feature

        output_mask = np.zeros((max_polyline, max_point), dtype=bool)
        output_mask[:num_polyline, :] = full_mask

        return output_feature, output_mask

    def global_to_frenet(self, traj_global: np.ndarray):
        '''
        This function converts the global coordinates to frenet coordinates
        args:
            traj_global: (N, 3) array, global coordinates [x, y, psi]
        return:
            traj_frenet: (N, 3) array, frenet coordinates [lanelet_id, s, d]
        '''
        if traj_global.ndim == 1:
            traj_global = traj_global[np.newaxis, :]

        num_waypoints = traj_global.shape[0]
        traj_frenet = np.zeros((num_waypoints, 3))

        for i, waypoint in enumerate(traj_global):
            lanelet, s, d = self.get_closest_lanelet(waypoint, check_psi=True)
            traj_frenet[i, :] = [lanelet.feature_id, s, d]

        return traj_frenet