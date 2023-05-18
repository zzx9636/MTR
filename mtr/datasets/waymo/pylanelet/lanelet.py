import warnings
import numpy as np
from typing import Union, Tuple, Dict
from .linestring import PyLineString
from collections.abc import Iterable


class PyLaneLet:

    def __init__(self, feature_id: int, centerline: PyLineString, type: int, speed_limit: float = 0) -> None:

        # Lanelet ID
        self.feature_id = feature_id

        # Lanelet type converted as a integer
        self.type = type

        # default to -1 if this information is not available
        self.speed_limit = speed_limit

        # road boundary and center line
        self.centerline = centerline

        # lanelet length (of center line)
        self.length = self.centerline.length

        self.predecessor = set()
        self.successor = set()

        self.merge_from = set()
        self.divide_to = set()

        self.left_changable = set()  # legally changable to left lane
        self.left_unchangable = set()  # legally unchangable to left lane
        self.right_changable = set()  # legally changable to right lane
        self.right_unchangable = set()  # legally unchangable to right lane

        self.num_points = self.centerline.points.shape[0]
        self.left_boundary = np.zeros((self.num_points, 2))
        self.right_boundary = np.zeros((self.num_points, 2))

        self.traffic_light = []
        self.stop_sign = []

        self.distance_to_nearest_agent = 0

    def get_bbox_vertices(self):
        '''
        Get vertices of the lane boundary, if the boundary is not defined as we return +-0.1 around the center line
        Returns:
            polygon_vertices: (2*num, 2) array of vertices, 
                        first half is left boundary, second half is right boundary (in reverse order)
        '''
        centerline_xy = self.centerline.points[:, :2]
        centerline_heading = self.centerline.points[:, 3:5]
        left_width = self.left_boundary[:, 0]

        left_width[left_width <= 0] = 0.5
        left_vertices = np.copy(centerline_xy)
        left_vertices[:, 0] = left_vertices[:, 0] - left_width * centerline_heading[:, 1]
        left_vertices[:, 1] = left_vertices[:, 1] + left_width * centerline_heading[:, 0]

        right_width = self.right_boundary[:, 0]
        right_width[right_width <= 0] = 0.5
        right_vertices = np.copy(centerline_xy)

        right_vertices[:, 0] = right_vertices[:, 0] + right_width * centerline_heading[:, 1]
        right_vertices[:, 1] = right_vertices[:, 1] - right_width * centerline_heading[:, 0]

        right_vertices_flip = np.flip(right_vertices, axis=0)
        return np.concatenate((left_vertices, right_vertices_flip), axis=0)

    def add_stop_sign(self, stop_sign: Dict):
        '''
        Append stop sign to the current lanelet
        We assume that the stop sign is at the end of the lanelet
        '''
        self.stop_sign.append(stop_sign)

    def add_traffic_light(self, traffic_light: Dict):
        '''
        This function append traffic light to the current lanelet
        '''
        self.traffic_light.append(traffic_light)

    def add_successor(self, successors: Union[list, int]):
        '''
        Add successor lanelet ID to the current lanelet
        Parameters:
            successors: int or list of int
        '''
        if isinstance(successors, int):
            self.successor.add(successors)
        else:
            self.successor.update(successors)

    def add_predecessor(self, predecessors: Union[list, int]):
        '''
        Add predecessor lanelet ID to the current lanelet
        Parameters:
            predecessors: int or list of int
        '''
        if isinstance(predecessors, int):
            self.predecessor.add(predecessors)
        else:
            self.predecessor.update(predecessors)

    def add_merge_from(self, merge_from: Union[list, int]):
        '''
        Add merge_from lanelet ID to the current lanelet
        Parameters:
            merge_from: int or list of int
        '''
        if isinstance(merge_from, int):
            self.merge_from.add(merge_from)
        else:
            self.merge_from.update(merge_from)

    def add_divide_to(self, divide_to: Union[list, int]):
        '''
        Add divide_to lanelet ID to the current lanelet
        Parameters:
            divide_to: int or list of int
        '''
        if isinstance(divide_to, int):
            self.divide_to.add(divide_to)
        else:
            self.divide_to.update(divide_to)

    def add_left(self, left: Union[list, int], changable: bool):
        '''
        Add left lanelet ID to the current lanelet
        Parameters:
            left: int or list of int
        '''
        target = self.left_changable if changable else self.left_unchangable
        if isinstance(left, int):
            target.add(left)
        else:
            target.update(left)

    def add_right(self, right: Union[list, int], changable: bool):
        '''
        Add right lanelet ID to the current lanelet
        Parameters:
            right: int or list of int
        '''
        target = self.right_changable if changable else self.right_unchangable
        if isinstance(right, int):
            target.add(right)
        else:
            target.update(right)

    def is_successor(self, lanelet_id: int):
        '''
        Check if the current lanelet is the successor of the lanelet with the given ID
        Parameters:
            lanelet_id: int
        '''
        return lanelet_id in self.successor

    def is_predecessor(self, lanelet_id: int):
        '''
        Check if the current lanelet is the predecessor of the lanelet with the given ID
        Parameters:
            lanelet_id: int
        '''
        return lanelet_id in self.predecessor

    def is_left(self, lanelet_id: int):
        '''
        Check if the current lanelet is the left lanelet of the lanelet with the given ID
        Parameters:
            lanelet_id: int
        '''
        return lanelet_id in self.left_changable or lanelet_id in self.left_unchangable

    def is_right(self, lanelet_id: int):
        '''
        Check if the current lanelet is the right lanelet of the lanelet with the given ID
        Parameters:
            lanelet_id: int
        '''
        return lanelet_id in self.right_changable or lanelet_id in self.right_unchangable

    def is_left_changable(self, lanelet_id: int):
        '''
        Check if the current lanelet is the left lanelet of the lanelet with the given ID
        '''
        return lanelet_id in self.left_changable

    def is_right_changable(self, lanelet_id: int):
        '''
        Check if the current lanelet is the right lanelet of the lanelet with the given ID
        '''
        return lanelet_id in self.right_changable

    def is_left_unchangable(self, lanelet_id: int):
        '''
        check if the current lanelet is the left lanelet of the lanelet with the given ID
        '''
        return lanelet_id in self.left_unchangable

    def is_right_unchangable(self, lanelet_id: int):
        '''
        check if the current lanelet is the left lanelet of the lanelet with the given ID
        '''
        return lanelet_id in self.right_unchangable

    def is_merge_from(self, lanelet_id: int):
        '''
        check if the current lanelet is the merge_from lanelet of the lanelet with the given ID
        '''
        return lanelet_id in self.merge_from

    def is_divide_to(self, lanelet_id: int):
        '''
        check if the current lanelet is the divide_to lanelet of the lanelet with the given ID
        '''
        return lanelet_id in self.divide_to

    def distance_to_centerline(self, point: np.ndarray):
        '''
        Get distance from the given point to the center line
        Parameters:
            point: (2,) array
        Returns:
            distance: float
            s: normalized position on the center line
        '''
        # This d is (curve(s) - point)
        s, d = self.centerline.spline.projectPoint(point)

        # sampled_pt = self.centerline.spline.getValue(s)
        deri = self.centerline.spline.getDerivative(s)
        slope = np.arctan2(deri[1], deri[0])  # [N,]
        return -np.sin(slope) * d[0] + np.cos(slope) * d[1], s

    @staticmethod
    def encode_polyline_type(
        lane_type: Union[np.ndarray, int, float],
        left_type: Union[np.ndarray, int, float],
        right_type: Union[np.ndarray, int, float],
    ) -> float:
        '''
        This function encodes the polyline 3D type into a single number
        '''
        # HACK: Hard code the hash table
        lane_type_hash = {
            0: 0,  # TYPE_UNKNOWN
            1: 1,  # TYPE_FREEWAY
            2: 2,  # TYPE_SURFACE_STREET
            3: 0,  # TYPE_BIKE_LANE
            17: 0,  # TYPE_STOP_SIGN
            18: 0,  # TYPE_CROSSWALK
            19: 0,  # TYPE_SPEED_BUMP
        }

        feature_type_hash = {
            0: 0,  # TYPE_UNKNOWN
            1: 0,  # TYPE_FREEWAY
            2: 0,  # TYPE_SURFACE_STREET
            3: 1,  # TYPE_BIKE_LANE
            17: 2,  # TYPE_STOP_SIGN
            18: 3,  # TYPE_CROSSWALK
            19: 4,  # TYPE_SPEED_BUMP
        }

        boundary_type_hash = {
            0: 0,  # TYPE_UNKNOWN
            6: 1,  #'TYPE_BROKEN_SINGLE_WHITE'
            7: 2,  #'TYPE_SOLID_SINGLE_WHITE'
            8: 3,  #'TYPE_SOLID_DOUBLE_WHITE'
            9: 4,  #'TYPE_BROKEN_SINGLE_YELLOW'
            10: 5,  #'TYPE_BROKEN_DOUBLE_YELLOW'
            11: 6,  #'TYPE_SOLID_SINGLE_YELLOW'
            12: 7,  #'TYPE_SOLID_DOUBLE_YELLOW'
            13: 8,  #'TYPE_PASSING_DOUBLE_YELLOW'
            15: 9,  #'TYPE_ROAD_EDGE_BOUNDARY'
            16: 10,  #'TYPE_ROAD_EDGE_MEDIAN'
        }
        factor = np.array([1, 3, len(boundary_type_hash) * 3])

        if isinstance(lane_type, Iterable):
            lane_type_hashed = np.array([lane_type_hash[i] for i in lane_type])
            feature_type_hashed = np.array([feature_type_hash[i] for i in lane_type])
        else:
            lane_type_hashed = lane_type_hash[lane_type]
            feature_type_hashed = feature_type_hash[lane_type]

        if isinstance(left_type, Iterable):
            left_type_hashed = np.array([boundary_type_hash[i] for i in left_type])
        else:
            left_type_hashed = boundary_type_hash[left_type]

        if isinstance(right_type, Iterable):
            right_type_hashed = np.array([boundary_type_hash[i] for i in right_type])
        else:
            right_type_hashed = boundary_type_hash[right_type]

        # We know that only TYPE_FREEWAY or TYPE_SURFACE_STREET can have left and right boundary
        stack_type_hashed = np.array([lane_type_hashed, left_type_hashed, right_type_hashed])  # (3,) or (3, N)
        type_hashed = np.dot(factor, stack_type_hashed) \
            + feature_type_hashed + (feature_type_hashed>0)*(3*(len(boundary_type_hash)**2)-1)
        return type_hashed

    def generate_feature(self, point_max: int = 512, append_stop_sign=False) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Create feature for the current lanelet
        Input: 
            point_max: int, maximum number of points in the polyline feature
            append_stop_sign: bool, whether to append stop sign feature to the end of the polyline feature
        Output:
            feature: (M, point_max, 11 ) array, 
                polyline feature [x, y, z, dx, dy, dz, speed limit, width_L, width_R, lane_type, feature_id]
            mask: (M, point_max) array, 1 for valid point, 0 for invalid point
        '''
        speed_limit = self.speed_limit * np.ones((self.centerline.points.shape[-2], 1))
        feature_id = self.feature_id * np.ones((self.centerline.points.shape[-2], 1))

        lane_type = self.encode_polyline_type(
            self.centerline.points[:, -1], self.left_boundary[:, -1], self.right_boundary[:, -1]
        )[:, np.newaxis]  #(N,1)

        boundary_width = np.concatenate(
            (self.left_boundary[:, 0][:, np.newaxis], self.right_boundary[:, 0][:, np.newaxis]), axis=1
        )  # (N, 2)

        # Get polyline feature
        polyline_feature = np.concatenate(
            (self.centerline.points[:, :-1], speed_limit, boundary_width, lane_type, feature_id), axis=1
        )

        if append_stop_sign:
            temp = [polyline_feature]
            for stop_sign in self.stop_sign:
                temp_feature = np.zeros((1, 11))  # 0 for unknown
                temp_feature[:, :3] = stop_sign['polyline'][:, :3]  # polyline in stop sign is 2D
                temp_feature[:, -2] = self.encode_polyline_type(
                    stop_sign['polyline'][:, -1], stop_sign['polyline'][:, -1] * 0, stop_sign['polyline'][:, -1] * 0
                )  # lane type
                temp_feature[:, -1] = stop_sign['id']
                temp.append(temp_feature)
            concat_feature = np.concatenate(temp, axis=0)  #[num_points+num_stopsign, 11]
            num_points = concat_feature.shape[-2]

            # generate full feature
            if num_points <= point_max:
                full_feature = np.zeros((1, point_max, 11))
                mask = np.zeros((1, point_max), dtype=bool)

                full_feature[0, :num_points, :] = concat_feature  #[1, point_max, 11]
                mask[0, :num_points] = True  #[1,point_max]
            else:
                warnings.warn(
                    f"Cannot include all polypoints. Current polyline have {num_points}, but max points is {point_max}."
                )
                full_feature = concat_feature[np.newaxis, -point_max:, :]  #[1, point_max, 11]
                mask = np.ones((1, point_max), dtype=bool)  #[1,point_max]
        else:
            num_stop_sign = len(self.stop_sign)
            full_feature = np.zeros((num_stop_sign + 1, point_max, 11))
            mask = np.zeros((num_stop_sign + 1, point_max), dtype=bool)

            num_points = polyline_feature.shape[0]
            if num_points <= point_max:
                full_feature[0, :num_points, :] = polyline_feature
                mask[0, :num_points] = True
            else:
                warnings.warn(
                    f"Cannot include all polypoints. Current polyline have {num_points}, but max points is {point_max}."
                )
                full_feature[0, :, :] = polyline_feature[-point_max:, :]
                mask[0, :] = True

            for i, stop_sign in enumerate(self.stop_sign):
                full_feature[i + 1, 0, :3] = stop_sign['polyline'][:, :3]  # polyline in stop sign is 2D
                full_feature[i + 1, 0, -4] = self.encode_polyline_type(
                    stop_sign['polyline'][:, -1], stop_sign['polyline'][:, -1] * 0, stop_sign['polyline'][:, -1] * 0
                )  # lane type
                full_feature[i + 1, 0, -1] = stop_sign['id']
                mask[i + 1, 0] = True

        return full_feature, mask
