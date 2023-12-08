# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for basic soft actor-critic.
"""

from typing import Optional, Union, Tuple, Dict
import os
import warnings
import torch
import copy
import numpy as np
from queue import PriorityQueue
import wandb
import time
from rl_env.waymax_env import MultiAgentEnvironment
from rl_env.env_utils import action_to_waymax_action

from rl.encoder import Encoder
from rl.actor import Actor
from rl.critic import Critic

from rl.rl_utils import ReplayMemory, collect_batch, to_device

class SAC():
  def __init__(
    self,
    cfg_solver,
    seed,
    data_iter: iter, 
    env: MultiAgentEnvironment,
    encoder: Encoder,  
    actor: Actor,
    ref_actor: Actor,
    critic: Critic,
  ):
    '''
    ############################# Load config
    self.cfg_solver = copy.deepcopy(cfg_solver)    
    
    self.device = torch.device(cfg_solver.device)
    # Training hyper-parameters.
    self.num_envs = int(cfg_solver.num_envs)
    self.max_steps = int(self.cfg_solver.max_steps)  # Maximum number of steps for training.
    self.opt_period = int(self.cfg_solver.opt_period)  # Optimizes actors/critics every `opt_period` steps.
    self.num_updates_per_opt = int(self.cfg_solver.num_updates_per_opt)  # The number of updates per optimization.
    self.eval_period = int(self.cfg_solver.eval_period)  # Evaluates actors/critics every `eval_period` steps.
    self.actor_update_period = int(self.cfg_solver.actor_update_period)  # Updates actor every `actor_update_period` steps.
    self.soft_update_period = int(self.cfg_solver.soft_update_period)  # Updates critic target every `soft_update_period` steps.
    self.soft_tau = float(self.cfg_solver.soft_tau)  # Soft update coefficient for the target network.
    self.warmup_steps = int(self.cfg_solver.warmup_steps)  # Uses random actions before `warmup_steps`.
    self.min_steps_b4_opt = int(self.cfg_solver.min_steps_b4_opt)  # Starts to optimize after `min_steps_b4_opt`.
    self.batch_size = int(cfg_solver.batch_size)
    self.max_model = int(cfg_solver.max_model) if cfg_solver.max_model is not None else None
    
    # Evaluation
    self.eval_b4_learn = bool(cfg_solver.eval.b4_learn)
    self.eval_metric: str = cfg_solver.eval.metric

    ########################## Replay Buffer.
    self.memory = ReplayMemory(int(cfg_solver.memory_capacity), seed)
    self.rng = np.random.default_rng(seed=seed)

    ######################### Logs checkpoints and visualizations.
    self.out_folder: str = self.cfg_solver.out_folder
    self.model_folder = os.path.join(self.out_folder, 'model')
    os.makedirs(self.model_folder, exist_ok=True)
    self.figure_folder = os.path.join(self.out_folder, 'figure')
    os.makedirs(self.figure_folder, exist_ok=True)
    self.use_wandb = bool(cfg_solver.use_wandb)
    if self.use_wandb:
      # initialize wandb
        wandb.init(entity='zzx9636', project='SAC')
    '''
    # Environment
    self.data_iter = data_iter
    self.env = env

    # Actor and Critic
    self.encoder = encoder
    self.actor = actor
    self.ref_actor = ref_actor
    self.critic = critic
    self.critic_target = copy.deepcopy(critic) 
  
  def learn(self):
    self.cnt_step: int = 0
    self.cnt_opt_period: int = 0
    self.cnt_eval_period: int = 0
    self.cnt_safety_violation: int = 0
    self.cnt_num_episode: int = 0
    
    start_learning = time.time()
    obsrv_all = None
    
    self.eval() 
    
    while self.cnt_step <= self.max_steps:
      # Interacts with the env and sample transitions.
      obsrv_all = self.interact(obsrv_all)

      # Optimizes actor and critic.
      self.update()
      
      self.eval()

    end_learning = time.time()
    time_learning = end_learning - start_learning
    print('\nLearning: {:.1f}'.format(time_learning))

  def interact(self, obs_cur: dict[str, torch.Tensor] = None):
    """
    Interacts with the environment by taking a step based on the current state.

    Args:
      cur_state: The current state of the environment.

    Returns:
      The next observation after taking a step in the environment.
    """
    if obs_cur is None:
      # sample a new episode
      scenario_id, cur_state = next(self.data_iter)
      with torch.no_grad():
        self.encoder.eval()
        cur_encoded_state, is_controlled = self.encoder(cur_state, None)
    else:
      scenario_id = obs_cur['scenario_id']
      cur_state = obs_cur['cur_state']
      cur_encoded_state = obs_cur['cur_encoded_state']
      is_controlled = obs_cur['is_controlled']
      
    
    self.actor.eval()
    with torch.no_grad():      
      # 1. Run a forward pass of the decoder to get action
      decoder_ouput = self.actor(cur_encoded_state)
      # 2. Sample Action  
      cur_action = self.actor.sample(decoder_ouput)['sample'].detach().cpu().numpy()
      
    # 3. Convert to action
    waymax_action = action_to_waymax_action(cur_action, is_controlled)
    
    # 4. Step the waymax environment
    next_state = self.env.step_sim_agent(cur_state, [waymax_action])
    
    # 5. Run metrics Function
    metrics = self.env.metrics(next_state, waymax_action)
    
    # Generate Rewards
    rewards = {}
    has_collision = False
    if 'overlap' in metrics.keys():
      #(num_agent,) # Min Distance to other agents
      rewards['overlap'] = np.asarray(metrics['overlap'].value[is_controlled].min(axis = -1))
      has_collision = np.any(rewards['overlap'] <= 0)
      
    has_offroad = False
    if 'offroad' in metrics.keys():
      # (num_agent,) # Distance to off-road area, off-road when negative
      rewards['offroad'] = np.asanyarray(-metrics['offroad'].value[is_controlled]) 
      has_offroad = np.any(rewards['offroad'] <= 0)
      
    has_infeasible_kinematics = False
    if 'kinematics' in metrics.keys():
      # (num_agent,) # Kinematics infeasibility when value is negative
      rewards['kinematics'] = np.asarray(-metrics['kinematics'].value[is_controlled]) 
      has_infeasible_kinematics = np.any(rewards['kinematics'] <= 0)
    
    g_x = np.minimum(rewards['overlap'], rewards['offroad'])
    g_x = np.minimum(g_x, rewards['kinematics'])
      
    # 6. Get the next observation  
    next_encoded_state, is_controlled = self.encoder(next_state, is_controlled)
    # print(has_collision, has_offroad, has_infeasible_kinematics, next_state.remaining_timesteps.item() <= 0)
    # 7. Check if the episode is done
    is_done = has_collision or has_offroad or has_infeasible_kinematics or next_state.remaining_timesteps.item() <= 0 
    n_agent = cur_action.shape[0]
    is_done_array = torch.ones(n_agent, dtype=torch.bool) * is_done
    
    # 8. create record
    record = {
      'scenario_id': to_device(scenario_id, 'cpu', detach=True),
      'cur_encoded_state': to_device(cur_encoded_state, 'cpu', detach=True),
      'cur_action': to_device(cur_action, 'cpu', detach=True),
      'rewards': to_device(rewards, 'cpu', detach=True), # Dict
      'g_x': to_device(g_x, 'cpu', detach=True), # (num_agent,)
      'next_encoded_state': to_device(next_encoded_state, 'cpu', detach=True), # Dict
      'is_done': to_device(is_done_array, 'cpu', detach=True), # (num_agent,)
      'is_controlled': to_device(is_controlled, 'cpu', detach=True)
    }
    
    # 9. save the record to the memory
    # self.memory.update(record)
    
    # 10. return the next observation
    if is_done:
      obs_next = None
    else:
      obs_next = {
        'cur_state': next_state,
        'cur_encoded_state': next_encoded_state,
        'is_controlled': is_controlled,
      }
    
    return obs_next
    
  
  def update_actor(self, record: Dict):
    """
    Update the actor network based on the given record.

    Args:
      record (Dict): A dictionary containing the current encoded state.

    Returns:
      Tuple: A tuple containing the loss values, alpha value, and batch size.
    """
        
    # Extract from the batch dict
    cur_encoded_state: Dict = record['cur_encoded_state']
        
    # update the model mode
    self.actor.train()
    self.ref_actor.eval()
    self.critic.eval()
    
    # 1. Run a forward pass of the decoder to get action
    decoder_ouput = self.actor(cur_encoded_state)
    
    # 2. Run a forward pass of the reference actor to get reference action distribution
    with torch.no_grad():
      ref_decoder_output = self.ref_actor(cur_encoded_state)
    
    ref_mode, ref_mix, ref_gmm = self.ref_actor.construct_distribution(ref_decoder_output)
    
    # 3. TODO: Compute the KL divergence
    
    # 4. Sample Action  
    sample = self.actor.sample(decoder_ouput)
    sampled_action = sample['sample']
    log_prob = sample['log_prob']
    
    batch_size = sampled_action.shape[0]
    
    # 5. Compute the Q value
    q1, q2 = self.critic(cur_encoded_state, sampled_action)
    
    # 4. update the actor
    loss_pi, loss_ent, loss_alpha = self.actor.update(q1=q1, q2=q2, log_prob=log_prob)
    
    return loss_pi, loss_ent, loss_alpha, batch_size
    
  def update_critics(self, batch: Dict) -> Tuple[float, float]:
    # Extract from the batch dict
    cur_encoded_state = batch['cur_encoded_state']
    cur_action = batch['cur_action']
    batch_size = cur_action.shape[0]
    g_x = batch['g_x']
    next_encoded_state = batch['next_encoded_state']
    done = batch['is_done']
    
    # update the model mode
    self.critic.train()
    self.critic_target.train()
    self.actor.eval()
    
    # sample next_action from the actor
    with torch.no_grad():      
      # 1. Run a forward pass of the decoder to get action
      decoder_ouput = self.actor(next_encoded_state)
      
      # 2. Sample Action  
      sample = self.actor.sample(decoder_ouput)
      
      next_action = sample['sample'].detach()
      next_log_prob = sample['log_prob'].detach()
      entropy_motives = -self.actor.alpha * next_log_prob
      
      assert batch_size == next_action.shape[0]
      
    # 3. Compute the target Q value
    q1_next, q2_next = self.critic_target(next_encoded_state, next_action)
    
    # 4. Compute the estimated Q value
    q1, q2 = self.critic(cur_encoded_state, cur_action)  # Gets Q(s, a).

    # 5. Update the critic
    loss_q = self.critic.update(
      q1=q1, q2=q2,
      q1_next=q1_next, q2_next=q2_next,
      done = done, reward=None,
      g_x=g_x, l_x=None,
      binary_cost=None,
      entropy_motives=entropy_motives,
    )
    
    return loss_q, batch_size
    
  def update(self):
    if (self.cnt_step >= self.min_steps_b4_opt and self.cnt_opt_period >= self.opt_period):
      print(f"Updates at sample step {self.cnt_step}")
      self.cnt_opt_period = 0
      loss_q_all = []
      loss_pi_all = []
      loss_ent_all = []
      loss_alpha_all = []

      critic_count = 0
      actor_count = 0
      
      for timer in range(self.num_updates_per_opt):
        sample = True
        cnt = 0
        while sample:
          batch = self.sample_batch()
          sample = torch.logical_not(torch.any(batch.non_final_mask))
          cnt += 1
          if cnt >= 10:
            break
        if sample:
          warnings.warn("Cannot get a valid batch!!", UserWarning)
          continue
        
        loss_q, batch_size = self.update_critics(batch)
        critic_count += batch_size
        loss_q_all.append(loss_q)
        self.critic.update_hyper_param()
        
        if timer % self.actor_update_period == 0:
          loss_pi, loss_ent, loss_alpha, batch_size = self.update_actor(batch)
          actor_count += batch_size
          loss_pi_all.append(loss_pi)
          loss_ent_all.append(loss_ent)
          loss_alpha_all.append(loss_alpha)
          self.actor.update_hyper_param()
          
        # do soft update of the critic target network
        if timer % self.soft_update_period == 0:
          self.critic_target.soft_update(self.critic, self.soft_tau)
          
      loss_q_mean = np.sum(loss_q_all)/critic_count
      loss_pi_mean = np.sum(loss_pi_all)/actor_count
      loss_ent_mean = np.sum(loss_ent_all)/actor_count
      loss_alpha_mean = np.sum(loss_alpha_all)/actor_count

      if self.use_wandb:
        log_dict = {
            "loss/critic": loss_q_mean,
            "loss/policy": loss_pi_mean,
            "loss/entropy": loss_ent_mean,
            "loss/alpha": loss_alpha_mean,
            # "metrics/cnt_safety_violation": self.cnt_safety_violation,
            # "metrics/cnt_num_episode": self.cnt_num_episode,
            "hyper_parameters/alpha": self.actor.alpha,
            "hyper_parameters/gamma": self.critic.gamma,
        }
        wandb.log(log_dict, step=self.cnt_step, commit=False)

  def eval(self):
    pass
  
  

