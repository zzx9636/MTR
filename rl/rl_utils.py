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
    return collect_batch([self.memory[i] for i in indices])

  def sample_recent(self, batch_size, recent_size):
    recent_size = min(len(self.memory), recent_size)
    indices = self.rng.integers(low=0, high=recent_size, size=(batch_size,))
    return collect_batch([self.memory[i] for i in indices])

  def __len__(self):
    return len(self.memory)

def collect_batch(batch_list: List, device: torch.device = 'cuda') -> Dict[str, torch.Tensor]:
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
    if type(value[0]) == dict:
      input_batch[key] = collect_batch(value, device)
    elif type(value[0]) == torch.Tensor:
      input_batch[key] = merge_batch_by_padding_2nd_dim(value).to(device)
    elif type(value[0]) == np.ndarray:
      input_batch[key] = torch.from_numpy(np.stack(value, axis=0)).to(device)
    elif type(value[0]) == list:
      # stack list of lists
      input_batch[key] = [item for sublist in value for item in sublist]
    else:
      input_batch[key] = value
  return input_batch
  
def to_device(obj: Union[Dict, torch.Tensor, np.ndarray], device: torch.device, detach = True) -> Union[Dict, torch.Tensor]:
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
  elif type(obj) == np.ndarray:
    return torch.from_numpy(obj.copy()).to(device)
  else:
    return obj


def merge_batch_by_padding_2nd_dim(tensor_list):
  if len(tensor_list[0].shape) > 1:
    maxt_feat0 = max([x.shape[1] for x in tensor_list])
    
    rest_size = tensor_list[0].shape[2:]
    
    ret_tensor_list = []
    for k in range(len(tensor_list)):
      cur_tensor = tensor_list[k]
      assert cur_tensor.shape[2:] == rest_size

      new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, *rest_size)
      new_tensor[:, :cur_tensor.shape[1], ...] = cur_tensor
      ret_tensor_list.append(new_tensor)
  else:
    ret_tensor_list = tensor_list
      
  return torch.cat(ret_tensor_list, dim=0).contiguous()  
