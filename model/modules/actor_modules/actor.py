""" Actor Policy implementation for use with my present packing environment. """
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
import torch

from model.modules.actor_modules.position_selection import PresentPositionActor
from model.modules.actor_modules.present_selection import PresentSelectionActor


class PresentActor(nn.Module):
    """ Policy nn for PresentEnv with spatial awareness """

    def __init__(self, present_list: list[list[torch.Tensor]], device=torch.device("cpu")):
        super().__init__()

        self.flatten = nn.Flatten()
        self.present_selection = PresentSelectionActor(present_list, device)
        self.position_selection = PresentPositionActor(device)

        _obs_to_present = TensorDictModule(
            self.present_selection,
            in_keys=["observation"],
            out_keys=["present_data"]
        )

        # _present_data_transform = TensorDictModule(

        # )

        self.device = device

    def forward(self, tensordict):
        """ Forward function for running of nn """

        batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1
        return TensorDict({
            "action": {
                "present_idx": {
                    "logits": None
                },
                "rot": {
                    "logits": None
                },
                "flip": {
                    "logits": None
                },
                "x": {
                    "loc": None,
                    "scale": None
                },
                "y": {
                    "loc": None,
                    "scale": None
                },
            },
        }, batch_size=torch.Size([batch_size]), device=self.device)
