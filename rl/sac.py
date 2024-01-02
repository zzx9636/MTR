# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for basic soft actor-critic.
"""

from typing import Tuple, Dict
import os
import torch
import copy
import numpy as np
from waymax import visualization
from waymax import config as waymax_config
import matplotlib.pyplot as plt
import wandb
import time
from rl_env.waymax_env import MultiAgentEnvironment
from rl_env.waymax_util import sample_to_action, WomdLoader
from rl_env.env_utils import smooth_scenario, inverse_unicycle_control

from rl.encoder import Encoder
from rl.actor import Actor
from rl.critic import Critic
from tqdm.auto import tqdm

from rl.rl_utils import ReplayMemory, collect_batch, to_device

class SAC():
  def __init__(
    self,
    cfg,
    seed,
    train_data_iter: WomdLoader, 
    val_data_iter: WomdLoader,
    env: MultiAgentEnvironment,
    encoder: Encoder,  
    actor: Actor,
    ref_actor: Actor,
    critic: Critic,
  ):
    
    ############################# Load config
    
    # Training hyper-parameters.
    self.max_steps = int(cfg.MAX_STEPS)  # Maximum number of steps for training.
    self.opt_period = int(cfg.OPT_PERIOD)  # Optimizes actors/critics every `opt_period` steps.
    self.num_updates_per_opt = int(cfg.NUM_UPDATES)  # The number of updates per optimization.
    self.eval_period = int(cfg.EVAL_PERIOD)  # Evaluates actors/critics every `eval_period` steps.
    self.eval_episodes = int(cfg.EVAL_EPISODES)  # The number of episodes for evaluation.
    self.actor_update_period = int(cfg.ACTOR_UPDATE_PERIOD)  # Updates actor every `actor_update_period` steps.
    self.soft_update_period = int(cfg.SOFT_UPDATE_PERIOD)  # Updates critic target every `soft_update_period` steps.
    self.soft_tau = float(cfg.SOFT_UPDATE_TAU)  # Soft update coefficient for the target network.
    self.min_steps_b4_opt = int(cfg.MIN_STEP_BEFORE_TRAIN)  # Starts to optimize after `min_steps_b4_opt`.
    self.batch_size = int(cfg.BATCH_SIZE)
    
    # Evaluation

    ########################## Replay Buffer.
    self.memory = ReplayMemory(int(cfg.MEMORY_SIZE), seed)
    self.rng = np.random.default_rng(seed=seed)

    ######################### Logs checkpoints and visualizations.
    self.out_folder: str = cfg.OUT_DIR
    self.model_folder = os.path.join(self.out_folder, 'model')
    os.makedirs(self.model_folder, exist_ok=True)
    self.figure_folder = os.path.join(self.out_folder, 'figure')
    os.makedirs(self.figure_folder, exist_ok=True)
    self.use_wandb = bool(cfg.USE_WANDB)
    if self.use_wandb:
      # initialize wandb
        wandb.init(entity='zzx9636', project='SAC')
    
    # Environment
    self.train_data_iter = train_data_iter
    self.val_data_iter = val_data_iter
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
    return None
    for i in tqdm(range(self.max_steps)):
      self.cnt_step = i
      self.cnt_opt_period += 1
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
    self.encoder.eval()
    self.actor.eval()
    
    if obs_cur is None:
      # sample a new episode
      scenario_id, scenario = self.train_data_iter.next()
      scenario = smooth_scenario(scenario)
      # Get the GT action
      gt_action, gt_action_valid = inverse_unicycle_control(scenario)
      # Reset the environment
      cur_state = self.env.reset(scenario)
      # Find the controlled agents
      is_controlled = self.encoder.is_controlled_func(cur_state)
      # Encode the state
      with torch.no_grad():
        cur_encoded_state, is_controlled = self.encoder(cur_state, is_controlled)
      scenario_id = [scenario_id] * is_controlled.sum().item()
    else:
      scenario_id = obs_cur['scenario_id']
      cur_state = obs_cur['cur_state']
      cur_encoded_state = obs_cur['cur_encoded_state']
      is_controlled = obs_cur['is_controlled']
          
    with torch.no_grad():      
      # 1. Run a forward pass of the decoder to get action
      decoder_ouput = self.actor(cur_encoded_state)
      # 2. Sample Action  
      cur_action = self.actor.sample(decoder_ouput)['sample'].detach().cpu().numpy()
      
    # 3. Convert to action
    waymax_action = sample_to_action(cur_action, is_controlled)
    
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
          
    g_x = np.minimum(rewards['overlap'], rewards['offroad'])
      
    # 6. Get the next observation  
    with torch.no_grad():
      is_controlled_next = self.encoder.is_controlled_func(next_state)
      next_encoded_state, _ = self.encoder(next_state, is_controlled_next)

    # 7. Check if the episode is done
    is_done = has_collision or has_offroad or next_state.remaining_timesteps.item() <= 0 
    n_agent = cur_action.shape[0]
    is_done_array = torch.ones(n_agent, dtype=torch.bool) * is_done
    
    # 8. create record
    record = {
      'scenario_id': scenario_id,
      'cur_encoded_state': to_device(cur_encoded_state, 'cpu', detach=True),
      'cur_action': to_device(cur_action, 'cpu', detach=True),
      'rewards': to_device(rewards, 'cpu', detach=True), # Dict
      'g_x': to_device(g_x, 'cpu', detach=True), # (num_agent,)
      'next_encoded_state': to_device(next_encoded_state, 'cpu', detach=True), # Dict
      'is_done': to_device(is_done_array, 'cpu', detach=True), # (num_agent,)
      'is_controlled': to_device(is_controlled_next, 'cpu', detach=True)
    }
    
    # 9. save the record to the memory
    self.memory.update(record)
    
    # 10. return the next observation
    if is_done:
      obs_next = None
    else:
      obs_next = {
        'scenario_id': scenario_id,
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
    # self.ref_actor.eval()
    self.critic.eval()
    
    # 1. Run a forward pass of the decoder to get action
    decoder_ouput = self.actor(cur_encoded_state)
    
    # 2. Run a forward pass of the reference actor to get reference action distribution
    # with torch.no_grad():
    #   ref_decoder_output = self.ref_actor(cur_encoded_state)
    
    # ref_mode, ref_mix, ref_gmm = self.ref_actor.construct_distribution(ref_decoder_output)
    
    # 3. TODO: Compute the KL divergence
    
    # 4. Sample Action  
    sample = self.actor.sample(decoder_ouput)
    sampled_action = sample['sample']
    log_prob = sample['log_prob']
    
    batch_size = sampled_action.shape[0]
    
    # 5. Compute the Q value
    q = self.critic(cur_encoded_state, sampled_action)
    
    # 4. update the actor
    loss_pi, loss_ent, loss_alpha = self.actor.update(q=q, log_prob=log_prob)
    
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
      q_next = self.critic_target(next_encoded_state, next_action)
    
    # 4. Compute the estimated Q value
    q = self.critic(cur_encoded_state, cur_action)  # Gets Q(s, a).

    # 5. Update the critic
    loss_q = self.critic.update(
      q = q,
      q_next=q_next,
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
        batch = self.memory.sample(self.batch_size)
        
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
      
      log_dict = {
            "loss/critic": loss_q_mean,
            "loss/policy": loss_pi_mean,
            "loss/entropy": loss_ent_mean,
            "loss/alpha": loss_alpha_mean,
            # "metrics/cnt_safety_violation": self.cnt_safety_violation,
            # "metrics/cnt_num_episode": self.cnt_num_episode,
            "hyper_parameters/alpha": self.actor.alpha.item(),
            "hyper_parameters/gamma": self.critic.gamma,
        }
      if self.use_wandb:
        wandb.log(log_dict, step=self.cnt_step, commit=False)
      else:
        print(log_dict)
 
  def eval(self):
    if self.cnt_step % self.eval_period != 0:
      return
    print('Evaluating the model at sample step {}'.format(self.cnt_step))
    self.val_data_iter.reset()
    
    eposide_length = []
    count_success = 0
    count_collision = 0
    count_offroad = 0
    
    self.encoder.eval()
    self.actor.eval()
    
    cur_figure_folder = os.path.join(self.figure_folder, f"step_{self.cnt_step}")
    os.makedirs(cur_figure_folder, exist_ok=True)
    
    for i, (scenario_id, scenario) in tqdm(enumerate(self.val_data_iter.iter), total=self.eval_episodes):
      if i >= self.eval_episodes:
        break
      cur_state = self.env.reset(scenario)
      cur_encoded_state, is_controlled = self.encoder(cur_state, None)
      while True:
        with torch.no_grad():
          # 1. Run a forward pass of the decoder to get action
          decoder_ouput = self.actor(cur_encoded_state)
          # 2. Sample Action  
          cur_action = self.actor.sample(decoder_ouput)['sample'].detach().cpu().numpy()
          
        # 3. Convert to action
        waymax_action = sample_to_action(cur_action, is_controlled)
        
        # 4. Step the waymax environment
        next_state = self.env.step_sim_agent(cur_state, [waymax_action])
        
        # 5. Run metrics Function
        metrics = self.env.metrics(next_state, waymax_action)
        
        has_collision = False
        if 'overlap' in metrics.keys():
          #(num_agent,) # Min Distance to other agents
          overlap = np.asarray(metrics['overlap'].value)[is_controlled].min(axis = -1)
          has_collision = np.any(overlap <= 0)
          
        has_offroad = False
        if 'offroad' in metrics.keys():
          # (num_agent,) # Distance to off-road area, off-road when negative
          offroad = np.asarray(-metrics['offroad'].value)[is_controlled]
          has_offroad = np.any(offroad <= 0)
                  
        done = False
        if has_collision:
          count_collision += 1
          done = True
        
        if has_offroad:
          count_offroad += 1
          done = True
        
        if next_state.remaining_timesteps.item() <= 0:
          count_success += 1
          done = True
          
        if done:
          eposide_length.append(next_state.timestep.item())
          # visualize 
          img = visualization.plot_simulator_state(
            next_state, use_log_traj=False, 
            highlight_obj = waymax_config.ObjectType.MODELED
          )
          plt.imshow(img)
          plt.axis('off')
          plt.title(f"Scenario {scenario_id}, Step {next_state.timestep.item()}, Offroad {has_offroad}, Collision {has_collision}")
          plt.savefig(os.path.join(cur_figure_folder, f"scenario_{scenario_id}.png"), 
                    bbox_inches='tight', pad_inches=0, dpi=300)
          plt.close()
          break
        else:
          with torch.no_grad():
            is_controlled = self.encoder.is_controlled_func(next_state)
            cur_encoded_state, _ = self.encoder(next_state, is_controlled)
            cur_state = next_state

    eval_result = {
      'avg_eposide_length': np.mean(eposide_length) - self.env.config.init_steps, 
      'success': count_success,
      'collision': count_collision,
      'offroad': count_offroad,
    }
    
    if self.use_wandb:
      wandb.log(eval_result, step=self.cnt_step, commit=False)
    else:
      print(eval_result)
    
  
  

