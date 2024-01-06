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
from torch.optim.lr_scheduler import StepLR

class Actor(nn.Module):
  def __init__(self, 
    cfg,
    actor_network: nn.Module,
    device: torch.device = 'cuda'
  ) -> None:
    nn.Module.__init__(self)
    
    self.cfg = cfg
    self.actor_type = cfg.ACTOR_TYPE # 'min' or 'max'
    self.kl_reg = cfg.KL_REG
    self.update_alpha = cfg.UPDATE_ALPHA
    
    # Load Network and kl Coefficient
    self.actor_network = actor_network
    self.actor_network.to(device)
    
    self.log_alpha = torch.tensor(
      [cfg.LOG_ALPHA_INIT],
      dtype=torch.float32,
      requires_grad=True,
      device=device
    )
    
    self.target_kl = torch.tensor(
      [cfg.TARGET_KL],
      dtype=torch.float32,
      device=device
    )
    
    self.build_optimizer(cfg)
  
  @property
  def alpha(self):
    return self.log_alpha.exp()

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
      self.actor_network.parameters(), lr=cfg.LR, weight_decay=0.01
    )
    self.optimizer_scheduler = StepLR(
      self.optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA
    )
    
    self.lr_end = cfg.LR_END

    # Optimizer for the temperature parameter.
    self.alpha_optimizer = AdamW([self.log_alpha], lr=cfg.ALPHA_LR, weight_decay=0.01)
    self.alpha_optimizer_scheduler = StepLR(
      self.alpha_optimizer, step_size=cfg.ALPHA_LR_PERIOD, gamma=cfg.ALPHA_LR_DECAY
    )
    self.alpha_lr_end = cfg.ALPHA_LR_END
      
  def update_hyper_param(self):
    if not self.eval:
      # Update learning rate for the actor network.
      lr = self.optimizer_scheduler.state_dict()['param_groups'][0]['lr']
      if lr <= self.lr_end:
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = self.lr_end
      else:
        self.optimizer_scheduler.step()
        
      # Update learning rate for the temperature parameter.
      lr = self.alpha_optimizer.state_dict()['param_groups'][0]['lr']
      if lr <= self.alpha_lr_end:
        for param_group in self.alpha_optimizer.param_groups:
          param_group['lr'] = self.alpha_lr_end
      else:
        self.alpha_optimizer_scheduler.step()

  def update(self, 
        q: torch.Tensor,
        kl_div: torch.Tensor,
      ) -> Tuple[float, float, float]:
    """
    Update the actor network based on the given Q-values, log probabilities, and alpha value.

    Args:
      q1 (torch.Tensor): The Q-values from the first critic network.
      log_prob (torch.Tensor): The log probabilities of the actions taken by the actor network.

    Returns:
      Tuple[float, float, float]: A tuple containing the loss values for Q evaluation, kl, and alpha.

    """
    # Update Actor
    if self.actor_type == 'min':
      q_pi = torch.max(q, dim=-1)[0]
    elif self.actor_type == 'max':
      q_pi = torch.min(q, dim=-1)[0]
      
    if self.kl_reg:
      loss_kl = self.alpha * kl_div.view(-1).mean()
    else:
      loss_kl = 0.0
      
    if self.actor_type == 'min':
      loss_q_eval = q_pi.mean()
    elif self.actor_type == 'max':
      loss_q_eval = -q_pi.mean()
    # print('loss_q_eval', loss_q_eval)
    # print('loss_kl', loss_kl)
    loss_pi = loss_q_eval + loss_kl
    
    self.optimizer.zero_grad()
    loss_pi.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
    self.optimizer.step()

    # Update Alpha
    if self.kl_reg and self.update_alpha:
      loss_alpha = (self.alpha * (-kl_div.detach() - self.target_kl))
      self.alpha_optimizer.zero_grad()
      loss_alpha.mean().backward()
      torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
      self.alpha_optimizer.step()
      #TODO: Do we need to clip alpha?
      # self.log_alpha.data = torch.min(self.log_alpha.data, self.init_alpha.data)
    else:
      loss_alpha = 0.0 * loss_q_eval

    return q_pi.sum().item(), kl_div.sum().item(), loss_alpha.sum().item()

  def forward(self, encoder_dict: Dict) -> Dict:
    """
    Forward pass through the motion decoder.

    Args:
        encoder_dict (dict): The batch dictionary.

    Returns:
        output_dict: The batch dictionary with the motion prediction.
    """ 
    return self.actor_network(encoder_dict)
  
  def construct_distribution(self, output_dict: Dict) -> Tuple:
    """
    Construct the distribution for the motion decoder.

    Args:
        output_dict (dict): The batch dictionary.

    Returns:
      mode: The Gaussian distribution of each mode.
      mix: The categorical distribution of the modes.
      gmm: The Gaussian mixture model.
    """ 
    pred_ctrls, pred_scores = output_dict['pred_list'][-1]
    mode, mix, gmm = self.actor_network.build_gmm_distribution(pred_ctrls, pred_scores)
    return mode, mix, gmm
  
  def sample(self, output_dict: Dict, best = False):
    return self.actor_network.sample(output_dict, best)
  
  def to(self, device):
    super().to(device)
    self.log_alpha = self.log_alpha.to(device)
    self.target_kl = self.target_kl.to(device)
    return self


