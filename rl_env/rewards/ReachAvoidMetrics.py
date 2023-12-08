# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for the Waymax environment."""
import jax
import jax.numpy as jnp

from waymax import config as _config
from waymax import datatypes
from waymax import metrics
from waymax.rewards import abstract_reward_function
from typing import Dict
from rl_env.rewards.overlap_metric import OverlapMetric
from rl_env.rewards.offroad_metric import OffroadMetric
from rl_env.rewards.kinematics_metric import KinematicsFeasibilityMetric

class ReachAvoidMetrics(abstract_reward_function.AbstractRewardFunction):
  """Metrics function that store metrics into a dictionary."""

  def __init__(self, config: _config.LinearCombinationRewardConfig):
    
    self._config = config
    self._run_overlap = 'overlap' in config.rewards.keys()
    self._run_offroad = 'offroad' in config.rewards.keys()
    self._run_kinemetics = 'kinematics' in config.rewards.keys()
    
    self.overlap_metric = OverlapMetric()
    self.offroad_metric = OffroadMetric()
    self.kinematics_metric = KinematicsFeasibilityMetric()
    
  def compute(
      self,
      simulator_state: datatypes.SimulatorState,
  ) -> Dict[str, jax.Array]:
    """Computes the reward as a linear combination of metrics.

    Args:
      simulator_state: State of the Waymax environment.
      
    Returns:
      An dictionary of metrics, where there is one reward per agent
      (..., num_objects).
    """

    reward_dict = {}
      
    # Run interaction metric if specified.
    if self._run_overlap:
      reward_dict['overlap'] = self.overlap_metric.compute(simulator_state) 
    
    if self._run_offroad:
      reward_dict['offroad'] = self.offroad_metric.compute(simulator_state)
      
    if self._run_kinemetics: 
      reward_dict['kinematics'] = self.kinematics_metric.compute(simulator_state)

    return reward_dict
