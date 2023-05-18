from pyspline import Curve
import numpy as np
from typing import List, Union
import warnings


class PyLineString:

    def __init__(
        self,
        feature_id: int,
        points: np.ndarray,
        feature_type: str = None,
        sub_type: str = None,
    ) -> None:
        '''
        Constructor for PyLineString
        :param id: ID of the line string
        :param points: Nx7 array of points (x,y,z,dir_x,dir_y,dir_z,type) in the line string
        '''
        self.feature_id = feature_id
        self.points = points
        try:
            points_cleaned = points[:, :3]
            _, idx_cleaned = np.unique(np.round(points_cleaned, 3), axis=0, return_index=True)
            points_cleaned = points_cleaned[np.sort(idx_cleaned), :]
            assert points_cleaned.shape[0] > 1, f'LineString {self.feature_id} has less than 2 unique points'
            self.spline = Curve(x=points_cleaned[:, 0], y=points_cleaned[:, 1], z=points_cleaned[:, 2], k=3)
            self.length = self.spline.getLength()
        except Exception as e:
            warnings.warn(f'Error when creating spline for LineString {self.feature_id}: {e}')
            self.spline = None
            self.length = 0
                
        self.feature_type = feature_type
        self.sub_type = sub_type
        self.related_ids = set()

    def __str__(self) -> str:
        return f'PyLineString {self.id} with type {self.feature_type} - {self.sub_type} and length {self.length}'

    def update_related_ids(self, ids: Union[List[int], int]) -> None:
        '''
        Update related IDs
        :param related_ids: list of related IDs
        '''
        if isinstance(ids, int):
            self.related_ids.add(ids)
        else:
            self.related_ids.update(ids)
            
    def resample(self, num_points):
        if self.spline is not None:
            self.points = self.sample_points(start = 0, end = 1, endpoint = True, num=num_points)

    def sample_points(self, start: float = 0, end: float = 1, endpoint: bool = False, num: int = 100) -> np.ndarray:
        '''
        Uniformly sample n points along the line string between start and end
        :param start: normalized start position [0,1)
        :param end: normalized end position (0,1]
        :param n: number of points to sample
        :return: Nxy array of points (x,y,z,dir_x,dir_y,dir_z,type)
        '''
        if self.spline is None:
            return self.points
        
        sampled_points = np.zeros((num, 7))
        
        s_list = np.linspace(start, end, num, endpoint=endpoint)
        sampled_points[:, :3] = self.spline.getValue(s_list)
        
        for i, s in enumerate(s_list):
            der = self.spline.getDerivative(s)
            sampled_points[i, 3:6] = der / np.linalg.norm(der)
            
        sampled_points[:,-1] = sampled_points[0,-1]

        return sampled_points
        
    def project_points(self, points: np.ndarray) -> float:
        '''
        Project a point onto the line string
        :param points: 2D point (x,y) or a Nx2 array of points
        :return: closest point(s) on the line string (Nx2 array) 
        '''
        if self.spline is None:
            return self.points

        s = self.get_s(points)
        projected_points = self.spline.getValue(s)
        return projected_points

    def distance_to_point(self, point: np.ndarray) -> float:
        '''
        Compute distance between a point and the line string
        :param point: 2D point (x,y) or a Nx2 array of points
        :return: distance(s) to the line string
        '''
        if self.spline is None:
            d = point - self.points[0, :3]
        else:
            point = point.astype(float)
            _, d = self.spline.projectPoint(point)
        return np.linalg.norm(d, axis=-1)

    def get_ref_pose(self, s):
        '''
        Get reference pose (x,y,theta) at normalized position s
        :param s: normalized position [0,1]
        :return: reference pose (x,y,theta)
        '''
        if self.spline is None:
            warnings.warn(f'LineString {self.feature_id} has less than 2 points, no reference pose can be computed')
            return np.array([self.points[0], self.points[1], 0])

        x, y = self.spline.getValue(s)
        deri = self.spline.getDerivative(s)
        theta = np.arctan2(deri[1], deri[0])
        return np.array([x, y, theta])