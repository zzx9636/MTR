from typing import Callable, Optional
import jax
import jax.numpy as jnp
from waymax import agents, datatypes
from waymax.agents import actor_core

from rl_env.env_utils import *

from typing import Dict, Tuple
from tools.mtr_lightning import MTR_Lightning
from rl.joint_policy import JointPolicy


_DEFAULT_CONTROL_FUNC = lambda state: state.object_metadata.is_modeled

@actor_core.register_actor_core
class SimAgentMTR(actor_core.WaymaxActorCore):
    def __init__(
        self,
        model_config,
        model_path: str,
        history_length: int = 11, 
        dt: float = 0.1,
        is_controlled_func: Optional[
            Callable[[datatypes.SimulatorState], jax.Array]
        ]= None,  
    ):
        super().__init__()
        if is_controlled_func is None:
            is_controlled_func = _DEFAULT_CONTROL_FUNC
        self.is_controlled_func = is_controlled_func
        
        # self.model = MTR_Lightning.load_from_checkpoint(model_path).cuda()
        self.model = JointPolicy(model_config)
        self.model.load_params_from_file(model_path)
        self.model.to('cuda')
        self.model.eval()
        
        self.history_length = history_length
        self.timestamps = np.arange(self.history_length) * dt
       
    def init(self, rng: jax.Array, state: datatypes.SimulatorState):
        """Returns an empty initial state."""
        raise NotImplementedError
    
    def select_action(
        self,
        params: actor_core.Params,
        state: datatypes.SimulatorState,
        actor_state: actor_core.ActorState,
        rng: jax.Array,
    ) -> agents.WaymaxActorOutput:
        """Selects an action given the current simulator state."""
        
        # actor_type = actor_state['actor_type']
        # del params, actor_state, rng
        
        is_controlled = self.is_controlled_func(state)
        # update the state
        state.object_metadata.is_modeled = is_controlled
        state.object_metadata.is_controlled = is_controlled
        input_dict = process_input(state, is_controlled)
        input_dict_batch = collate_batch([input_dict])
        
        # Do a forward pass
        # # ! hard code
        # batch_decoder_mapping = {
        #     'agent': [i for i in range(is_controlled.sum())]
        # }
        
        with torch.no_grad():
            output = self.model(input_dict_batch, None)
        
        actions_sampled = self.model.sample(output)['agent']['sample'].detach().cpu().numpy()
        
        actions_array = np.zeros((is_controlled.shape[0], 3))
        actions_array[is_controlled] = actions_sampled
        actions_valid = jnp.asarray(is_controlled[...,None])
        
        actions = datatypes.Action(data=jnp.asarray(actions_array), valid=actions_valid)
        
        return actor_core.WaymaxActorOutput(
            action=actions,
            actor_state=None,
            is_controlled=is_controlled,
        )
            
    @property
    def name(self) -> str:
        return 'mtr'
    
    
    