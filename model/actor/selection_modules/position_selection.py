""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from tensordict import TensorDict
from torch import nn
import torch


class PresentPositionActor(nn.Module):
    """ Policy nn for PresentEnv to choose where to place present. """

    def __init__(self, presents, device, grid_features=256, present_features=64):
        super().__init__()

        # Presents shouldn't be learnable/modifiable nor in state_dict
        self.register_buffer("presents", presents, persistent=False)
        self.device = device
        self.input_features = grid_features + present_features

        self.x_mean = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        ).to(self.device)
        self.x_std = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        ).to(self.device)
        self.y_mean = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        ).to(self.device)
        self.y_std = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        ).to(self.device)

    def forward(self, tensordict):
        """ Choose a placement position for the selected present in present_orient """
        present_idx = tensordict.get(("orient_data", "present_idx"))
        orient_idx = tensordict.get(("orient_data", "orient_idx"))
        orients = tensordict.get(("orient_data", "orients"))
        orient_mask = tensordict.get(("orient_mask", "chosen_orient"))

        orient = orients[orient_mask]
        orient_features = orient.get("orient_features")
        modulated_grid = orient.get("modulated_grid")

        # Concat present and modulated grid together
        combined_features = torch.cat([orient_features, modulated_grid], dim=1)

        x_mean = self.x_mean(combined_features)
        x_std = self.x_std(combined_features)
        y_mean = self.y_mean(combined_features)
        y_std = self.y_std(combined_features)

        # get present for partial action output
        present = self.presents[:, present_idx,  # type: ignore
                                orient_idx, :, :]

        return TensorDict({
            "action": {
                "present_idx": present_idx,
                "present": present
            },
            "pos_probs": {
                "x": {
                    "loc": x_mean,
                    "scale": x_std
                },
                "y": {
                    "loc": y_mean,
                    "scale": y_std
                }
            }
        })
