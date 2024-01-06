import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Network
import jax
jax.config.update('jax_platform_name', 'cpu')

from jax import numpy as jnp
import torch
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np

from mtr.models.context_encoder.mtr_encoder import MTREncoder
from mtr.models.motion_decoder.bc_decoder import BCDecoder
from mtr.models.motion_decoder.q_decoder import QDecoder
from rl.actor import Actor
from rl.critic import Critic
from rl.encoder import Encoder
from rl.sac import SAC

import dataclasses
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics
from rl_env.waymax_env import  MultiAgentEnvironment
from rl_env.waymax_util import WomdLoader
from rl_env.unicycle_model import InvertibleUnicycleModel
import copy

from mtr.config import cfg, cfg_from_yaml_file


max_num_objects = 32

# create a dataset
WOMD_1_2_0_VAL = _config.DatasetConfig(
    path='/Data/Dataset/Waymo/V1_2_tf/validation_interactive/validation_interactive_tfexample.tfrecord@150',
    # path='/Data/Dataset/Waymo/V1_2_tf/validation/validation_tfexample.tfrecord@150',
    max_num_rg_points=30000,
    data_format=_config.DataFormat.TFRECORD,
    max_num_objects=max_num_objects,
    shuffle_seed = 0,
    num_shards=4,
)

WOMD_1_2_0_TRAIN = _config.DatasetConfig(
    path='/Data/Dataset/Waymo/V1_2_tf/training/training_tfexample.tfrecord@1000',
    max_num_rg_points=30000,
    data_format=_config.DataFormat.TFRECORD,
    max_num_objects=max_num_objects,
    shuffle_seed = 0,
)

data_iter_train = WomdLoader(data_config=WOMD_1_2_0_TRAIN)
data_iter_val = WomdLoader(data_config=WOMD_1_2_0_VAL)


# Config the multi-agent environment:
init_steps = 1

# Set the dynamics model the environment is using.
# Note each actor interacting with the environment needs to provide action
# compatible with this dynamics model.
dynamics_model = InvertibleUnicycleModel()

# Expect users to control all valid object in the scene.
env = MultiAgentEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        _config.EnvironmentConfig(),
        init_steps = init_steps,
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.MODELED,
        rewards = _config.LinearCombinationRewardConfig(
            rewards={
                'overlap': 1.0, # Positive is good.  
                'offroad': -1.0, # Positive is good after the negative sign.
            }
        )
    ),
)


@jax.jit
def check_controlled(state: datatypes.SimulatorState):
    current_stamp = state.timestep
    is_valid = state.sim_trajectory.valid[..., current_stamp]
    
    is_modeled = jnp.logical_or(
        state.object_metadata.is_modeled,
        state.object_metadata.is_sdc
    )
    
    # is_modeled = is_valid
    # is_modeled.at[10:].set(False)
    is_vehicle = (state.object_metadata.object_types == 1)
    
    return jnp.logical_and(is_valid, jnp.logical_and(is_modeled, is_vehicle))

cfg = cfg_from_yaml_file('tools/cfgs/waymo/rl_finetune.yaml', cfg)
cfg.SAC.RL.EVAL_EPISODES = 10
cfg.SAC.ACTOR.ENTROPY_REG = False
cfg.SAC.ACTOR.UPDATE_ALPHA = False
model_dict = torch.load('output/bc_bicycle_4_freeze/epoch=0-step=755000.ckpt')['state_dict']

encoder_network = MTREncoder(cfg.MODEL.CONTEXT_ENCODER)

actor_network = BCDecoder(
    encoder_network.num_out_channels,
    cfg.MODEL.MOTION_DECODER
)

critic_network = QDecoder(
    encoder_network.num_out_channels,
    cfg.MODEL.Q_DECODER
)

encoder = Encoder(
    model_cfg=cfg.MODEL.CONTEXT_ENCODER,
    is_controlled_func = check_controlled,
)
encoder.context_encoder.load_model(model_dict)
actor = Actor(
    cfg.SAC.ACTOR,
    actor_network,
)
actor.actor_network.load_model(model_dict)
actor.eval()

critic = Critic(
    cfg.SAC.CRITIC,
    critic_network
)   

sac = SAC(
    cfg = cfg.SAC.RL ,
    seed = 0,
    train_data_iter = data_iter_train,
    val_data_iter = data_iter_val,
    env = env,
    encoder = encoder,
    actor = actor,
    ref_actor = copy.deepcopy(actor),
    critic = critic,
)

sac.learn()