from typing import Dict, List
import numpy as np
import torch
from collections import deque

class ReplayMemory(object):

  def __init__(self, capacity, seed):
    self.reset(capacity)
    self.capacity = capacity
    self.seed = seed
    self.rng = np.random.default_rng(seed=self.seed)

  def reset(self, capacity):
    if capacity is None:
      capacity = self.capacity
    self.memory = deque(maxlen=capacity)

  def update(self, transition):
    self.memory.appendleft(transition)  # pop from right if full

  def sample(self, batch_size):
    length = len(self.memory)
    indices = self.rng.integers(low=0, high=length, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def sample_recent(self, batch_size, recent_size):
    recent_size = min(len(self.memory), recent_size)
    indices = self.rng.integers(low=0, high=recent_size, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def __len__(self):
    return len(self.memory)


def collect_batch(batch_list: List, device: torch.device) -> Dict[str, torch.Tensor]:
  """Collects a batch of data from a list of transitions.

  Args:
      batch_list (List): a list of transitions.
      device (torch.device): device to store the data.

  Returns:
      Dict[str, torch.Tensor]: a batch of data.
  """
  pass