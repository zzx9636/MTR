# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Dict
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import StepLR
from rl.rl_utils import StepLRMargin

class Critic(nn.Module):
  def __init__(self, 
    cfg,
    q_network: nn.Module,
    device: torch.device = 'cuda'
  ) -> None:
    nn.Module.__init__(self)
    
    self.cfg = cfg
    self.mode = cfg.MODE # 'performance' or 'safety' or 'reach-avoid' or 'risk'
    self.update_gamma = cfg.UPDATE_GAMMA
    self.kl_reg = cfg.KL_REG
    
    self.q_network = q_network
    
    self.to(device)
    
    self.build_optimizer(cfg)
  
  @property
  def gamma(self):
    if self.update_gamma:
      return self.gamma_scheduler.value
    else:
      return self.cfg.GAMMA

  def build_optimizer(self, cfg):
    '''
    Build optimizers for the actor network and the temperature parameter.

    Args:
      cfg (Config): Configuration object containing hyperparameters.

    Returns:
      None
    '''
    # Optimizer for the actor network.
    self.optimizer = AdamW(
      self.parameters(), lr=cfg.LR, weight_decay=0.01
    )
    self.optimizer_scheduler = StepLR(
      self.optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA
    )
    self.lr_end = cfg.LR_END

    # Discount Factor Scheduler
    self.gamma_scheduler = StepLRMargin(
      init_value=cfg.GAMMA, 
      period=cfg.GAMMA_PERIOD,
      decay=cfg.GAMMA_DECAY, 
      goal_value=cfg.GAMMA_END,
      end_value=None,
    )
        
  def update_hyper_param(self):
    if not self.eval:
      # Update learning rate for the actor network.
      lr = self.optimizer_scheduler.state_dict()['param_groups'][0]['lr']
      if lr <= self.lr_end:
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = self.lr_end
      else:
        self.optimizer_scheduler.step()
        
      # update the discount factor
      if self.update_gamma:
        self.gamma_scheduler.step()

  def forward(self, encoder_output: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    '''
    Forward pass of the critic network.

    Args:
      encoder_output (Dict[str, torch.Tensor]): Output from the encoder network.
      action: (torch.Tensor): Action taken by the actor network.

    Returns:
      q1 (torch.Tensor): [num_agent, num_q], Output from the critic network.
    '''
    q = self.q_network(encoder_output, action)
    return q

  def update(
      self, 
      q: torch.Tensor,
      q_next: torch.Tensor,
      done: torch.Tensor, 
      reward: torch.Tensor, 
      g_x: torch.Tensor, 
      l_x: torch.Tensor,
      binary_cost: torch.Tensor, 
      kl_motives: torch.Tensor
  ) -> float:
    """Updates critic network with next Q values (target).

    Args:
        q1 (torch.Tensor):
        q2 (torch.Tensor):
        q1_next (torch.Tensor):
        q2_next (torch.Tensor):
        done (torch.Tensor):
        reward (torch.Tensor):
        g_x (torch.Tensor):
        l_x (torch.Tensor):
        binary_cost (torch.Tensor):
        kl_motives (torch.Tensor):

    Returns:
        float: critic loss.
    """
    done = done.float()
    # Gets Bellman update.
    y = self.get_bellman_update(
      q_next=q_next,
      done=done,
      reward=reward,
      g_x=g_x,
      l_x=l_x,
      binary_cost=binary_cost
    )
    
    if self.kl_reg:
      y += (1.0-done)*self.gamma * kl_motives
    # Repeat y
    y = y.unsqueeze(-1).repeat(1, q.shape[-1])
    # Regresses MSE loss for both Q1 and Q2.
    loss = mse_loss(input=q, target=y, reduction='none')
    # print(y)
    # print(q)
    # Backpropagates.
    self.optimizer.zero_grad()
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
    self.optimizer.step()
    return loss.sum().item()
  
  def get_bellman_update(
    self,
    q_next: torch.Tensor,
    done: torch.Tensor,
    reward: torch.Tensor = None,
    g_x: torch.Tensor = None,
    l_x: torch.Tensor = None,
    binary_cost: torch.Tensor = None,
  ):
    """
    Calculates the Bellman update for the critic network.

    Args:
      q_next (torch.Tensor): [num_agent, num_q] The Q-values from the first critic network for the next state.
      done (torch.Tensor): A tensor indicating whether the episode has terminated (1 if terminated, 0 otherwise).
      reward (torch.Tensor, optional): The reward tensor. Only used in 'performance' mode. Defaults to None.
      g_x (torch.Tensor, optional): The g(s): signed distance to failure set. Only used in 'reach-avoid' and 'safety' modes. Defaults to None.
      l_x (torch.Tensor, optional): The l(s): signed distance to the goal set. Only used in 'reach-avoid' mode. Defaults to None.
      binary_cost (torch.Tensor, optional): The binary cost tensor. Only used in 'risk' mode. Defaults to None.
    Returns:
      torch.Tensor: The updated Q-values. [num_agent, 1]

    Raises:
      ValueError: If an unsupported RL mode is provided.
    """
    
    # Conservative target Q values: if the control policy
    # we want to learn is to maximize, we take the minimum of the two
    # Q values. Otherwise, we take the maximum.
    assert len(q_next.shape) == 2, "q_next should be [num_agent, num_q]"
    if self.mode == 'risk':
      target_q = torch.max(q_next, dim=-1)[0]
    elif (self.mode == 'reach-avoid' or self.mode == 'safety' or self.mode == 'performance'):
      target_q = torch.min(q_next, dim=-1)[0]
    else:
      raise ValueError("Unsupported RL mode.")
    
    if self.mode == 'reach-avoid':
      gamma = self.gamma * (1-done)
      # V(s) = min{ g(s), max{ l(s), V(s') }}
      # Q(s, u) = V( f(s,u) ) = main g(s'), max{ ell(s'), min_{u'} Q(s', u')}}
      terminal_target = torch.min(l_x, g_x)
      original_target = torch.min(g_x, torch.max(l_x, target_q))
      y = (1.0-gamma) * terminal_target + gamma*original_target 
    elif self.mode == 'safety':
      gamma = self.gamma * (1-done)
      # V(s) = min{ g(s), V(s') }
      # Q(s, u) = V( f(s,u) ) = min{ g(s'), max_{u'} Q(s', u') }
      # normal state
      y = (1.0-gamma) * g_x + gamma*torch.min(g_x, target_q) 
    elif self.mode == 'performance':
      y = reward + self.gamma * target_q * (1-done)
    elif self.mode == 'risk':
      y = binary_cost + self.gamma * target_q * (1-done)
    return y

  def soft_update(self, target_model, tau):
    """Soft update model parameters.

    Args:
        target_model (Critic): Target model to update.
        tau (float): Soft update coefficient.
    """
    for target_param, param in zip(target_model.parameters(), self.parameters()):
      target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    