'''Metrics to calculate the signed distance between objects.'''

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry
import numpy as np
from typing import Optional, Tuple

''''
Adapt from https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/geometry_utils.py
'''
@jax.jit
def minkowski_sum_of_box_and_box_points(box1_points: jax.Array,
                                        box2_points: jax.Array) -> jax.Array:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

    The last dimensions of the input and return store the x and y coordinates of
    the points. Both box1_points and box2_points needs to be stored in
    counter-clockwise order. Otherwise the function will return incorrect results
    silently.

    Args:
        box1_points: Tensor of vertices for box 1, with shape:
        (num_boxes, num_points_per_box, 2).
        box2_points: Tensor of vertices for box 2, with shape:
        (num_boxes, num_points_per_box, 2).

    Returns:
        The Minkowski sum of the two boxes, of size (num_boxes,
        num_points_per_box * 2, 2). The points will be stored in counter-clockwise
        order.
    """
    # NUM_BOX_1 = box1_points.shape[0]
    # NUM_BOX_2 = box2_points.shape[0]
    NUM_VERTICES_IN_BOX = box1_points.shape[1]
    assert NUM_VERTICES_IN_BOX == 4, "Only support boxes"
    # Hard coded order to pick points from the two boxes. This is a simplification
    # of the generic convex polygons case. For boxes, the adjacent edges are
    # always 90 degrees apart from each other, so the index of vertices can be
    # hard coded.
    point_order_1 = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
    point_order_2 = jnp.array([0, 1, 1, 2, 2, 3, 3, 0])
    
    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(
        box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(
        box2_points)
    
    # The cross-product of the unit vectors indicates whether the downmost edge
    # in box2 is pointing to the left side (the inward side of the resulting
    # Minkowski sum) of the downmost edge in box1. If this is the case, pick
    # points from box1 in the order `point_order_2`, and pick points from box2 in
    # the order of `point_order_1`. Otherwise, we switch the order to pick points
    # from the two boxes, pick points from box1 in the order of `point_order_1`,
    # and pick points from box2 in the order of `point_order_2`.
    # Shape: (num_boxes, 1)
    condition = (
        jnp.cross(downmost_box1_edge_direction, downmost_box2_edge_direction)
        >= 0.0
    )
    # box1_point_order of size [num_boxes, num_points_per_box * 2 = 8, 1].
    box1_point_order = jnp.where(condition, point_order_2, point_order_1)
    box1_point_order = jnp.expand_dims(box1_point_order, axis=-1)
    
    # Shift box1_point_order by box1_start_idx, so that the first index in
    # box1_point_order is the downmost vertex in the box.
    box1_point_order = jnp.mod(box1_point_order + box1_start_idx,
                                    NUM_VERTICES_IN_BOX)
    # Gather points from box1 in order.
    # ordered_box1_points is of size [num_boxes, num_points_per_box * 2, 2].
    ordered_box1_points = jnp.take_along_axis(box1_points, box1_point_order, axis=-2)

    # Gather points from box2 as well.
    box2_point_order = jnp.where(condition, point_order_1, point_order_2)
    box2_point_order = jnp.expand_dims(box2_point_order, axis=-1)
    box2_point_order = jnp.mod(box2_point_order + box2_start_idx,
                                    NUM_VERTICES_IN_BOX)
    ordered_box2_points = jnp.take_along_axis(box2_points, box2_point_order, axis=-2)

    minkowski_sum = ordered_box1_points + ordered_box2_points
    return minkowski_sum

@jax.jit
def _get_downmost_edge_in_box(box: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Finds the downmost (lowest y-coordinate) edge in the box.

    Note: We assume box edges are given in a counter-clockwise order, so that
    the edge which starts with the downmost vertex (i.e. the downmost edge) is
    uniquely identified.

    Args:
    box: (num_boxes, num_points_per_box, 2). The last dimension contains the x-y
        coordinates of corners in boxes.

    Returns:
    A tuple of two tensors:
        downmost_vertex_idx: The index of the downmost vertex, which is also the
        index of the downmost edge. Shape: (num_boxes, 1, 1).
        downmost_edge_direction: The tangent unit vector of the downmost edge,
        pointing in the counter-clockwise direction of the box.
        Shape: (num_boxes, 1, 2).
    """
    # The downmost vertex is the lowest in the y dimension.
    # Shape: (num_boxes, 1).
    
    NUM_BOX, NUM_VERTICES_IN_BOX, _ = box.shape
    assert NUM_VERTICES_IN_BOX == 4, "Only support boxes"
    downmost_vertex_idx = jnp.argmin(box[..., 1], axis=-1)[..., None, None]

    # Find the counter-clockwise point edge from the downmost vertex.
    edge_start_vertex = jnp.take_along_axis(box, downmost_vertex_idx, axis=1)
    # edge_start_vertex = box[np.arange(NUM_BOX), downmost_vertex_idx, :]
    edge_end_idx = jnp.mod(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end_vertex = jnp.take_along_axis(box, edge_end_idx, axis=1)
    # edge_end_vertex = box[np.arange(NUM_BOX), edge_end_idx, :]

    # Compute the direction of this downmost edge.
    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = jnp.linalg.norm(downmost_edge, axis=-1, keepdims=True)
    downmost_edge_direction = downmost_edge / downmost_edge_length
    return downmost_vertex_idx, downmost_edge_direction

@jax.jit
def _get_edge_info(
    polygon_points: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes properties about the edges of a polygon.

    Args:
        polygon_points: Tensor containing the vertices of each polygon, with
        shape (num_polygons, num_points_per_polygon, 2). Each polygon is assumed
        to have an equal number of vertices.

    Returns:
        tangent_unit_vectors: A unit vector in (x,y) with the same direction as
        the tangent to the edge. Shape: (num_polygons, num_points_per_polygon, 2).
        normal_unit_vectors: A unit vector in (x,y) with the same direction as
        the normal to the edge.
        Shape: (num_polygons, num_points_per_polygon, 2).
        edge_lengths: Lengths of the edges.
        Shape (num_polygons, num_points_per_polygon).
    """
    # Shift the polygon points by 1 position to get the edges.
    # Shape: (num_polygons, 1, 2).
    first_point_in_polygon = polygon_points[:, 0:1, :]
    # Shape: (num_polygons, num_points_per_polygon, 2).
    shifted_polygon_points = jnp.concatenate(
        [polygon_points[:, 1:, :], first_point_in_polygon], axis=-2)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    edge_vectors = shifted_polygon_points - polygon_points

    # Shape: (num_polygons, num_points_per_polygon).
    edge_lengths = jnp.linalg.norm(edge_vectors, axis=-1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    tangent_unit_vectors = edge_vectors / jnp.expand_dims(edge_lengths, axis=-1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    normal_unit_vectors = jnp.stack(
        [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], axis=-1)
    return tangent_unit_vectors, normal_unit_vectors, edge_lengths

@jax.jit
def signed_distance_from_point_to_convex_polygon(
    query_points: jax.Array, 
    polygon_points: jax.Array
) -> jax.Array:
    """Finds the signed distances from query points to convex polygons.

    Each polygon is represented by a 2d tensor storing the coordinates of its
    vertices. The vertices must be ordered in counter-clockwise order. An
    arbitrary number of pairs (point, polygon) can be batched on the 1st
    dimension.

    Note: Each polygon is associated to a single query point.

    Args:
        query_points: (2). The last dimension is the x and y
        coordinates of points.
        polygon_points: (batch_size, num_points_per_polygon, 2). The last
        dimension is the x and y coordinates of vertices.

    Returns:
        A tensor containing the signed distances of the query points to the
        polygons. Shape: (batch_size,).
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(polygon_points)

    # Expand the shape of `query_points` to (num_polygons, 1, 2), so that
    # it matches the dimension of `polygons_points` for broadcasting.
    # query_points = query_points[None, None, :]
    query_points = jnp.expand_dims(query_points, axis=(0, 1))
    
    # Compute query points to polygon points distances.
    # Shape (num_polygons, num_points_per_polygon, 2).
    vertices_to_query_vectors = query_points - polygon_points
    
    # Shape (num_polygons, num_points_per_polygon).
    vertices_distances = jnp.linalg.norm(vertices_to_query_vectors, axis=-1)

    # Query point to edge distances are measured as the perpendicular distance
    # of the point from the edge. If the projection of this point on to the edge
    # falls outside the edge itself, this distance is not considered (as there)
    # will be a lower distance with the vertices of this specific edge.

    # Make distances negative if the query point is in the inward side of the
    # edge. Shape: (num_polygons, num_points_per_polygon).
    edge_signed_perp_distances = jnp.sum(
        -normal_unit_vectors * vertices_to_query_vectors, axis=-1)

    # If `edge_signed_perp_distances` are all less than 0 for a
    # polygon-query_point pair, then the query point is inside the convex polygon.
    is_inside = jnp.all(edge_signed_perp_distances <= 0, axis=-1)

    # Project the distances over the tangents of the edge, and verify where the
    # projections fall on the edge.
    # Shape: (num_polygons, num_edges_per_polygon).
    projection_along_tangent = jnp.sum(
        tangent_unit_vectors * vertices_to_query_vectors, axis=-1)
    projection_along_tangent_proportion = projection_along_tangent/edge_lengths
    
    # Shape: (num_polygons, num_edges_per_polygon).
    is_projection_on_edge = jnp.logical_and(
        projection_along_tangent_proportion >= 0.0,
        projection_along_tangent_proportion <= 1.0)

    # If the point projection doesn't lay on the edge, set the distance to inf.
    edge_perp_distances = jnp.abs(edge_signed_perp_distances)
    edge_distances = jnp.where(is_projection_on_edge,
                                edge_perp_distances, np.inf)

    # Aggregate vertex and edge distances.
    # Shape: (num_polyons, 2 * num_edges_per_polygon).
    edge_and_vertex_distance = jnp.concatenate([edge_distances, vertices_distances],
                                        axis=-1)
    # Aggregate distances per polygon and change the sign if the point lays inside
    # the polygon. Shape: (num_polygons,).
    min_distance = jnp.min(edge_and_vertex_distance, axis=-1)
    signed_distances = jnp.where(is_inside, -min_distance, min_distance)
    return signed_distances

@jax.jit 
def filter_signed_distance(signed_distance, valid):
    num_obj = signed_distance.shape[0]
    # Remove self-interaction
    i, j = jnp.diag_indices(num_obj)
    signed_distance = signed_distance.at[i, j].set(1e3)
    
    # Remove Invalid objects
    valid = jnp.outer(valid, valid) # Shape: (num_objects, num_objects)
    valid = valid * ~jnp.eye(num_obj, dtype=jnp.bool_) # Shape: (num_objects, num_objects)
    signed_distance = jnp.where(valid, signed_distance, 1e3)
    return signed_distance
    
class OverlapMetric(abstract_metric.AbstractMetric):
    """Overlap metric.
    
    This metric returns negative if an object's bounding box is overlapping with
    that of another object.
    """

    @jax.named_scope('InteractionMetric.compute')
    def compute(
        self, simulator_state: datatypes.SimulatorState
    ) -> abstract_metric.MetricResult:
        current_object_state = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )
        return self.compute_overlap(current_object_state)
    
    def compute_overlap(
        self, current_traj: datatypes.Trajectory
    ) -> abstract_metric.MetricResult:
        """Computes the interaction metric.
    
        Args:
        current_traj: Trajectory object containing current states of shape (...,
            num_objects, num_timesteps=1).
    
        Returns:
        A (num_objects, num_objects) MetricResult of pairwise distance.
        """
        traj_5dof = current_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw']).squeeze(-2)
        num_obj = traj_5dof.shape[0]

        # Shape: (num_objects, 5)
        
        # Get corners of the bounding boxes
        # Shape: (num_objects, 4, 2)
        corners = geometry.corners_from_bboxes(traj_5dof)
        corners_all = jnp.expand_dims(corners, axis = 1).repeat(num_obj, axis=1) # Shape: (num_objects, num_objects, 4, 2)
        corners_all_transpose = corners_all.transpose((1, 0, 2, 3)) # Shape: (num_objects, num_objects, 4, 2)
        
        corners_all = corners_all.reshape(-1, 4, 2)
        corners_all_transpose = corners_all_transpose.reshape(-1, 4, 2)
        
        minkowski_diff = minkowski_sum_of_box_and_box_points(corners_all, -corners_all_transpose)
        signed_distance = signed_distance_from_point_to_convex_polygon(np.array([0,0]), minkowski_diff) # Shape: (num_objects * num_objects, )
        signed_distance = signed_distance.reshape(num_obj, num_obj)
        
        
        # Remove self-interaction
        self_interaction = jnp.eye(num_obj, dtype=jnp.bool_)
        signed_distance = jnp.where(self_interaction, 1e3, signed_distance)
        
        # Remove Invalid objects
        valid = current_traj.valid.squeeze(-1) # Shape: (num_objects,)
        valid = jnp.outer(valid, valid) # Shape: (num_objects, num_objects)
        valid = valid * ~self_interaction # Shape: (num_objects, num_objects)
        signed_distance = jnp.where(valid, signed_distance, 1e3)
                        
        return abstract_metric.MetricResult.create_and_validate(
            value = jnp.asarray(signed_distance), valid = valid
        )    