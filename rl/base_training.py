# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for training agents.

This file implements a parent class for all training agents, modified from
https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/base_training.py
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Callable
from queue import PriorityQueue
import time
import os
import copy
import wandb
import numpy as np
import torch
from rl.rl_utils import ReplayMemory, collect_batch

from agent.base_block import build_network
from simulators.vec_env.vec_env import VecEnvBase
from simulators import BaseEnv


class BaseTraining(ABC):
  leaderboard: Union[PriorityQueue, np.ndarray]

  def __init__(self, cfg_solver, cfg_arch, seed: int):
    self.cfg_solver = copy.deepcopy(cfg_solver)

    self.device = torch.device(cfg_solver.device)
    self.rollout_devce = self.cfg_solver.rollout_env_device
    self.critics, self.actors = build_network(cfg_solver, cfg_arch, self.device, verbose=True)

    # Training hyper-parameters.
    self.num_envs = int(cfg_solver.num_envs)
    self.max_steps = int(self.cfg_solver.max_steps)  # Maximum number of steps for training.
    self.opt_period = int(self.cfg_solver.opt_period)  # Optimizes actors/critics every `opt_period` steps.
    self.num_updates_per_opt = int(self.cfg_solver.num_updates_per_opt)  # The number of updates per optimization.
    self.eval_period = int(self.cfg_solver.eval_period)  # Evaluates actors/critics every `eval_period` steps.
    self.warmup_steps = int(self.cfg_solver.warmup_steps)  # Uses random actions before `warmup_steps`.
    self.min_steps_b4_opt = int(self.cfg_solver.min_steps_b4_opt)  # Starts to optimize after `min_steps_b4_opt`.
    self.batch_size = int(cfg_solver.batch_size)
    self.max_model = int(cfg_solver.max_model) if cfg_solver.max_model is not None else None

    # Replay Buffer.
    self.memory = ReplayMemory(int(cfg_solver.memory_capacity), seed)
    self.rng = np.random.default_rng(seed=seed)

    # Evaluation
    self.eval_b4_learn = bool(cfg_solver.eval.b4_learn)
    self.eval_metric: str = cfg_solver.eval.metric

    # Logs checkpoints and visualizations.
    self.out_folder: str = self.cfg_solver.out_folder
    self.model_folder = os.path.join(self.out_folder, 'model')
    os.makedirs(self.model_folder, exist_ok=True)
    self.figure_folder = os.path.join(self.out_folder, 'figure')
    os.makedirs(self.figure_folder, exist_ok=True)
    self.use_wandb = bool(cfg_solver.use_wandb)

  def sample_batch(self, batch_size: Optional[int] = None, recent_size: int = 0):
    if batch_size is None:
      batch_size = self.batch_size
    if recent_size > 0:  # use recent
      list_batch = self.memory.sample_recent(batch_size, recent_size)
    else:
      list_batch = self.memory.sample(batch_size)
    # TODO: Determine where to put collect_batch
    return collect_batch(list_batch, self.device)

  @abstractmethod
  def interact(
      self, obsrv_all: Optional[dict[str, torch.Tensor]],
  ):
    raise NotImplementedError

  @abstractmethod
  def update(self):
    raise NotImplementedError

  @abstractmethod
  def update_hyper_param(self):
    raise NotImplementedError

  @abstractmethod
  def eval(self, env, rollout_env, eval_callback, init_eval: bool = False) -> bool:
    raise NotImplementedError

  @abstractmethod
  def save(self):
    raise NotImplementedError

  def learn(self, env: BaseEnv, eval_callback: Optional[Callable] = None):
    rollout_env = self.init_learn(env)

    if self.eval_b4_learn and eval_callback is not None:
      self.eval(env, rollout_env, eval_callback, init_eval=True)

    start_learning = time.time()
    obsrv_all = None 
    
    while self.cnt_step <= self.max_steps:
      # Interacts with the env and sample transitions.
      obsrv_all = self.interact(obsrv_all)

      # Optimizes actor and critic.
      self.update()

      # Evaluates after fixed number of gradient updates.
      if eval_callback is not None:
        eval_flag = self.eval(env, rollout_env, eval_callback)
        if eval_flag:  # Resets anyway.
          obsrv_all = None

      # Updates gamma, lr, etc.
      self.update_hyper_param()

    end_learning = time.time()
    time_learning = end_learning - start_learning
    print('\nLearning: {:.1f}'.format(time_learning))
    wandb.log({'time_learning': time_learning})

    # Saves the final actor and critic anyway.
    self.save()

    loss_record = np.array(self.loss_record, dtype=float)
    eval_record = np.array(self.eval_record)
    violation_record = np.array(self.violation_record)
    episode_record = np.array(self.episode_record)
    return (loss_record, eval_record, violation_record, episode_record, self.leaderboard)

  def init_learn(self, env: BaseEnv) -> Union[BaseEnv, VecEnvBase]:

    # Placeholders for training records.
    self.loss_record = []
    self.eval_record = []
    self.violation_record = []
    self.episode_record = []
    self.cnt_step: int = 0
    self.cnt_opt_period: int = 0
    self.cnt_eval_period: int = 0
    self.cnt_safety_violation: int = 0
    self.cnt_num_episode: int = 0

    if self.num_envs > 1:
      rollout_env = VecEnvBase([copy.deepcopy(env) for _ in range(self.num_envs)], device=self.rollout_devce)
      rollout_env.seed(env.seed_val)
      self.agent_copy_list = [copy.deepcopy(env.agent) for _ in range(self.num_envs)]
      for agent in self.agent_copy_list:
        agent.policy.net.to(torch.device(self.rollout_devce))
      rollout_env.set_attr("agent", self.agent_copy_list, value_batch=True)
    else:
      rollout_env = copy.deepcopy(env)
      rollout_env.seed(env.seed_val + 1)

    return rollout_env

  def restore(
      self, model_folder: str, actors_step: Dict[str, int], critics_step: Dict[str, int], verbose: bool = True
  ):
    """Restores the models from the given folder and steps.

    Args:
        model_folder (str): the path to the models, under this folder there should be folders with `net_name` of all
            actors and critics.
        actors_step (Dict[int]): a dictionary consists of the step of the checkpoint to load for each actor.
        critics_step (Dict[int]): a dictionary consists of the step of the checkpoint to load for each critic.
        verbose (bool, optional): print info if True. Defaults to True.
    """
    for actor_name, actor in self.actors.items():
      actor.restore(actors_step[actor_name], model_folder, verbose=verbose)

    for critic_name, critic in self.critics.items():
      critic.restore(critics_step[critic_name], model_folder, verbose=verbose)
