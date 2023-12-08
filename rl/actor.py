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
    device: torch.device = torch.device('cuda')
  ) -> None:
    nn.Module.__init__(self)
    
    self.cfg = cfg
    self.actor_type = cfg.ACTOR_TYPE # 'min' or 'max'
    self.entropy_reg = cfg.ENTROPY_REG
    self.update_alpha = cfg.UPDATE_ALPHA
    
    # Load Network and Entropy Coefficient
    self.actor_network = actor_network
    self.actor_network.to(device)
    
    self.log_alpha = torch.tensor(
      [cfg.LOG_ALPHA_INIT],
      dtype=torch.float32,
      requires_grad=True,
    )
    
    self.target_entropy = torch.tensor(
      [cfg.TARGET_ENTROPY],
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
        q1: torch.Tensor,
        q2: torch.Tensor,
        log_prob: torch.Tensor,
      ) -> Tuple[float, float, float]:
    """
    Update the actor network based on the given Q-values, log probabilities, and alpha value.

    Args:
      q1 (torch.Tensor): The Q-values from the first critic network.
      q2 (torch.Tensor): The Q-values from the second critic network.
      log_prob (torch.Tensor): The log probabilities of the actions taken by the actor network.

    Returns:
      Tuple[float, float, float]: A tuple containing the loss values for Q evaluation, entropy, and alpha.

    """
    # Update Actor
    if self.actor_type == 'min':
      q_pi = torch.max(q1, q2)
    elif self.actor_type == 'max':
      q_pi = torch.min(q1, q2)
      
    if self.entropy_reg:
      loss_entropy = self.alpha * log_prob.view(-1).mean()
    else:
      loss_entropy = 0.0
      
    if self.actor_type == 'min':
      loss_q_eval = q_pi.mean()
    elif self.actor_type == 'max':
      loss_q_eval = -q_pi.mean()

    loss_pi = loss_q_eval + loss_entropy
    
    self.optimizer.zero_grad()
    loss_pi.backward()
    self.optimizer.step()

    # Update Alpha
    if self.entropy_reg and self.update_alpha:
      loss_alpha = (self.alpha * (-log_prob.detach() - self.target_entropy)).mean()
      self.alpha_optimizer.zero_grad()
      loss_alpha.backward()
      self.alpha_optimizer.step()
      #TODO: Do we need to clip alpha?
      # self.log_alpha.data = torch.min(self.log_alpha.data, self.init_alpha.data)

    return {
      'loss_q_eval': loss_q_eval.item(),
      'loss_entropy': loss_entropy.item(),
      'loss_alpha': loss_alpha.item()
    }

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
  
  def sample(self, output_dict):
    """
    Sample a trajectory from the motion decoder.

    Args:
        batch_dict (dict): The batch dictionary.

    Returns:
        output_dict: The batch dictionary with the sampled trajectory added.
    """
    mode, mix, gmm = self.construct_distribution(output_dict)
    
    # sample_action = gmm.sample()
    mode: torch.distributions.MultivariateNormal
    mix: torch.distributions.Categorical
    
    # Sample from all Gaussian
    sample_all = mode.rsample() # [Batch, M, 3]
    sample_all_log_prob = mode.log_prob(sample_all)
    
    sample_mode = mix.sample() # [Batch]
    sample_mode_log_prob = mix.log_prob(sample_mode)
    
    sample_action = torch.gather(
        sample_all, 
        1, 
        sample_mode.unsqueeze(-1).unsqueeze(-1).repeat_interleave(sample_all.shape[-1], dim=-1)
    ).squeeze(-2)
    
    sample_action_log_prob = torch.gather(
        sample_all_log_prob, 
        1, 
        sample_mode.unsqueeze(-1)
    ).squeeze(-1)  + sample_mode_log_prob
                
    sample = sample_action * self.actor_network.output_std + self.actor_network.output_mean
    
    sample_dict = {
        'sample': sample,
        'log_prob': sample_action_log_prob
    }
        
    return sample_dict

  def sample_best(self, output_dict):
      """
      Sample a trajectory from the motion decoder.

      Args:
          batch_dict (dict): The batch dictionary.

      Returns:
          output_dict: The batch dictionary with the sampled trajectory added.
      """
  
      cur_decoder = self.actor_network
      
      pred_ctrls, pred_scores = output_dict['pred_list'][-1]
      
      best_idx = torch.argmax(pred_scores, dim=-1)
      
      # take value from the best index
      sample = pred_ctrls[torch.arange(pred_ctrls.shape[0]), best_idx, :3]
      
      sample = sample * cur_decoder.output_std + cur_decoder.output_mean
      
      sample_dict = {'sample': sample}
      
      return sample_dict

  def to(self, device):
    super().to(device)
    self.log_alpha = self.log_alpha.to(device)
    self.target_entropy = self.target_entropy.to(device)
    return self


