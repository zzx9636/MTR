# Import Network
import jax
from jax import numpy as jnp
from mtr.models.context_encoder.mtr_encoder import MTREncoder
from mtr.models.motion_decoder.bc_decoder import BCDecoder
from mtr.models.motion_decoder.q_decoder import QDecoder
from rl.actor import Actor
from rl.critic import Critic
from rl.encoder import Encoder
from rl.sac import SAC
import torch
import dataclasses
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics
from rl_env.waymax_env import  MultiAgentEnvironment
from rl_env.env_utils import WomdLoader
import tensorflow as tf
import copy

from mtr.config import cfg, cfg_from_yaml_file

@jax.jit
def check_controlled(state: datatypes.SimulatorState):
    current_stamp = state.timestep
    is_valid = state.sim_trajectory.valid[..., current_stamp]
    
    is_modeled = jnp.logical_or(
        state.object_metadata.is_modeled,
        state.object_metadata.is_sdc
    )
    
    is_vehicle = (state.object_metadata.object_types == 1)
    
    return jnp.logical_and(is_valid, jnp.logical_and(is_modeled, is_vehicle))
    
if __name__ == "__main__":
    max_num_objects = 32

    # create a dataset
    WOMD_1_2_0_VAL_LOCAL = _config.DatasetConfig(
        path='/Data/Dataset/Waymo/V1_2_tf/validation_interactive/validation_interactive_tfexample.tfrecord@150',
        max_num_rg_points=30000,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
        shuffle_seed = 0,
        repeat = 1,
    )

    WOMD_1_2_0_TRAIN_LOCAL = _config.DatasetConfig(
        path='/Data/Dataset/Waymo/V1_2_tf/training/training_tfexample.tfrecord@1000',
        max_num_rg_points=30000,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=max_num_objects,
        shuffle_seed = 0,
    )
    train_data_iter = WomdLoader(data_config=WOMD_1_2_0_TRAIN_LOCAL)
    val_data_iter = WomdLoader(data_config=WOMD_1_2_0_VAL_LOCAL)


    # Config the multi-agent environment:
    init_steps = 11

    # Set the dynamics model the environment is using.
    # Note each actor interacting with the environment needs to provide action
    # compatible with this dynamics model.
    dynamics_model = dynamics.DeltaLocal()

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
                    'offroad': 1.0, # Negative is good.
                    # 'kinematics': 1.0, # Negative is good.
                }
            )
        ),
    )
    
    cfg = cfg_from_yaml_file('tools/cfgs/waymo/rl_finetune.yaml', cfg)
    model_dict = torch.load('output/bc_atten_4_unfreeze_state/epoch=19-step=1211340.ckpt', map_location=torch.device('cpu'))['state_dict']
    
    encoder_network = MTREncoder(cfg.MODEL.CONTEXT_ENCODER)
    encoder_network.load_model(model_dict)

    actor_network = BCDecoder(
        encoder_network.num_out_channels,
        cfg.MODEL.MOTION_DECODER
    )
    actor_network.load_model(model_dict)
    critic_network = QDecoder(
        encoder_network.num_out_channels,
        cfg.MODEL.Q_DECODER
    )

    encoder = Encoder(
        model_cfg=cfg.MODEL.CONTEXT_ENCODER,
        is_controlled_func = check_controlled,
    )

    actor = Actor(
        cfg.SAC.ACTOR,
        actor_network,
    )
    actor.eval()

    critic = Critic(
        cfg.SAC.CRITIC,
        critic_network
    )   

    sac = SAC(
        cfg = cfg.SAC.RL ,
        seed = 0,
        train_data_iter = train_data_iter,
        val_data_iter = val_data_iter,
        env = env,
        encoder = encoder,
        actor = actor,
        ref_actor = None, #copy.deepcopy(actor),
        critic = critic,
    )

    sac.learn()