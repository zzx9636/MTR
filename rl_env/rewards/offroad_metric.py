'''Metrics to calculate the signed distance of road map.'''


import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric

class OffroadMetric(abstract_metric.AbstractMetric):
  """Offroad metric.

  This metric returns signed distance of nearest road section if the object is offroad.
  """

  @jax.named_scope('OffroadMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    """Computes the offroad metric.

    Args:
      simulator_state: Updated simulator state to calculate metrics for. Will
        compute the offroad metric for timestep `simulator_state.timestep`.

    Returns:
      An array containing the metric result of the same shape as the input
        trajectories. The shape is (..., num_objects).
    """
    current_object_state = datatypes.dynamic_slice(
        simulator_state.sim_trajectory,
        simulator_state.timestep,
        1,
        -1,
    )
    offroad = distance_offroad(current_object_state, simulator_state.roadgraph_points)
    
    valid = current_object_state.valid[..., 0]

    return abstract_metric.MetricResult.create_and_validate(
        offroad, valid
    )


def distance_offroad(
    trajectory: datatypes.Trajectory,
    roadgraph_points: datatypes.RoadgraphPoints,
) -> jax.Array:
  """Checks if the given trajectory is offroad.

  This determines the signed distance between each bounding box corner and the
  closest road edge (median or boundary). If the distance is negative, then the
  trajectory is onroad else offroad.

  Args:
    trajectory: Agent trajectories to test to see if they are on or off road of
      shape (..., num_objects). The bounding boxes derived from center and shape
      of the trajectory will be used to determine if any point in the box is
      offroad.
    roadgraph_points: All of the roadgraph points in the run segment of shape
      (..., num_points). Roadgraph points of type `ROAD_EDGE_BOUNDARY` and
      `ROAD_EDGE_MEDIAN` are used to do the check.

  Returns:
    min_distances: a float array with the shape (..., num_objects). The object is offroad 
    if the value is positive.
  """
  # Shape: (..., num_objects, num_corners=4, 2).
  bbox_corners = trajectory.bbox_corners[..., 0, :]
  # Add in the Z dimension from the current center. This assumption will help
  # disambiguate between different levels of the roadgraph (i.e. under and over
  # passes).
  # Shape: (..., num_objects, 1, 1).
  z = jnp.ones_like(bbox_corners[..., 0:1]) * trajectory.z[..., jnp.newaxis, :]
  # Shape: (..., num_objects, num_corners=4, 3).
  bbox_corners = jnp.concatenate((bbox_corners, z), axis=-1)
  shape_prefix = bbox_corners.shape[:-3]
  num_agents, num_points, dim = bbox_corners.shape[-3:]
  # Shape: (..., num_objects * num_corners=4, 3).
  bbox_corners = jnp.reshape(
      bbox_corners, [*shape_prefix, num_agents * num_points, dim]
  )
  # Here we compute the signed distance between the given trajectory and the
  # roadgraph points. The shape prefix represents a set of batch dimensions
  # denoted above as (...). Here we call a set of nested vmaps for each of the
  # batch dimensions in the shape prefix to allow for more flexible parallelism.
  compute_fn = compute_signed_distance_to_nearest_road_edge_point
  for _ in shape_prefix:
    compute_fn = jax.vmap(compute_fn)

  # Shape: (..., num_objects * num_corners=4).
  distances = compute_fn(bbox_corners, roadgraph_points)
  # Shape: (..., num_objects, num_corners=4).
  distances = jnp.reshape(distances, [*shape_prefix, num_agents, num_points])
  min_distances = jnp.min(distances, axis=-1)
  # Shape: (..., num_objects).
  return min_distances


def compute_signed_distance_to_nearest_road_edge_point(
    query_points: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    z_stretch: float = 2.0,
) -> jax.Array:
  """Computes the signed distance from a set of queries to roadgraph points.

  Args:
    query_points: A set of query points for the metric of shape
      (num_query_points, 3).
    roadgraph_points: A set of roadgraph points of shape (num_points).
    z_stretch: Tolerance in the z dimension which determines how close to
      associate points in the roadgraph. This is used to fix problems with
      overpasses.

  Returns:
    Signed distances of the query points with the closest road edge points of
      shape (num_query_points). If the value is negative, it means that the
      actor is on the correct side of the road, if it is positive, it is
      considered `offroad`.
  """
  # Shape: (..., num_points, 3).
  sampled_points = roadgraph_points.xyz
  # Shape: (num_query_points, num_points, 3).
  differences = sampled_points - query_points[:, jnp.newaxis]
  # Stretch difference in altitude to avoid over/underpasses.
  z_stretched_differences = differences * jnp.array([[[1.0, 1.0, z_stretch]]])
  square_distances = jnp.sum(z_stretched_differences**2, axis=-1)
  # Do not consider invalid points.
  # Shape: (num_points).
  is_road_edge = datatypes.is_road_edge(roadgraph_points.types)
  square_distances = jnp.where(
      roadgraph_points.valid & is_road_edge, square_distances, float('inf')
  )
  # Shape: (num_query_points).
  nearest_indices = jnp.argmin(square_distances, axis=-1)
  prior_indices = jnp.maximum(
      jnp.zeros_like(nearest_indices), nearest_indices - 1
  )
  nearest_xys = sampled_points[nearest_indices, :2]
  # Direction of the road edge at the nearest points. Should be normed and
  # tangent to the road edge.
  # Shape: (num_points, 2).
  nearest_vector_xys = roadgraph_points.dir_xyz[nearest_indices, :2]
  # Direction of the road edge at the points that precede the nearest points.
  # Shape: (num_points, 2).
  prior_vector_xys = roadgraph_points.dir_xyz[prior_indices, :2]
  # Shape: (num_query_points, 2).
  points_to_edge = query_points[..., :2] - nearest_xys
  # Get the signed distance to the half-plane boundary with a cross product.
  cross_product = jnp.cross(points_to_edge, nearest_vector_xys)
  cross_product_prior = jnp.cross(points_to_edge, prior_vector_xys)
  # If the prior point is contiguous, consider both half-plane distances.
  # Shape: (num_points).
  prior_point_in_same_curve = jnp.equal(
      roadgraph_points.ids[nearest_indices], roadgraph_points.ids[prior_indices]
  )
  offroad_sign = jnp.sign(
      jnp.where(
          jnp.logical_and(
              prior_point_in_same_curve, cross_product_prior < cross_product
          ),
          cross_product_prior,
          cross_product,
      )
  )
  # Shape: (num_query_points).
  return (
      jnp.linalg.norm(nearest_xys - query_points[:, :2], axis=-1) * offroad_sign
  )
