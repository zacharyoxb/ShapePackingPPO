""" Datatype to store saved model data """
from dataclasses import dataclass

import torch


@dataclass
class ModelData:
    """ Layout of saved / checkpointed model data """
    avg_reward: torch.Tensor
    policy_state: dict
    value_state: dict
    loss_state: dict
    optim_state: dict
    scheduler_state: dict
