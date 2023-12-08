from typing import Dict, List, Union
import numpy as np
import torch
from collections import deque

class StepLRMargin():
  def __init__(
      self, init_value, period, goal_value, decay=0.1, end_value=None,
      last_epoch=-1, threshold=0
  ):
    self.cnt = last_epoch
    self.value = None
    self.init_value = init_value
    self.period = period
    self.decay = decay
    self.end_value = end_value
    self.goal_value = goal_value
    self.threshold = threshold
    self.step()
  
  def step(self):
    self.cnt += 1
    self.value = self.get_value()
    
  def get_value(self):
    cnt = self.cnt - self.threshold
    if cnt < 0:
      return self.init_value

    numDecay = int(cnt / self.period)
    tmpValue = self.goal_value - (self.goal_value
                                  - self.init_value) * (self.decay**numDecay)
    if self.end_value is not None and tmpValue >= self.end_value:
      return self.end_value
    return tmpValue
  
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
  
def to_device(obj: Union[Dict, torch.Tensor], device: torch.device, detach = True) -> Union[Dict, torch.Tensor]:
  """Moves a dict or a tensor to a device.

  Args:
      obj (Union[Dict, torch.Tensor]): a dict or a tensor.
      device (torch.device): device to store the data.

  Returns:
      Union[Dict, torch.Tensor]: a dict or a tensor on the device.
  """
  if type(obj) == dict:
    return {k: to_device(v, device, detach) for k, v in obj.items()}
  elif type(obj) == torch.Tensor:
    if detach:
      return obj.detach().to(device)
    else:
      return obj.to(device)
  else:
    return obj
  