""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from tensordict import TensorDict
from torch import nn
import torch


class PresentPositionActor(nn.Module):
    """ Policy nn for PresentEnv to choose where to place present. """

    def __init__(self, presents, device, grid_features, present_features):
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

    def forward(self, choice_tensor, orient_td):
        """ Choose a placement position for the selected present in orients """
        orient_idx = torch.argmax(choice_tensor, dim=1)
        orient = orient_td.get("orients")[:, orient_idx.item()]

        present_idx = orient.get("present_idx")
        orient_idx = orient.get("orient_idx")
        orient_features = orient.get("orient_features")
        modulated_grid = orient.get("modulated_grid")

        # Concat present and modulated grid together
        combined_features = torch.cat([
            torch.flatten(orient_features, start_dim=1),
            torch.flatten(modulated_grid, start_dim=1)
        ], dim=1)

        x_mean = self.x_mean(combined_features)
        x_std = self.x_std(combined_features)
        y_mean = self.y_mean(combined_features)
        y_std = self.y_std(combined_features)

        # get present for partial action output
        present = self.presents[  # type: ignore
            present_idx.item(), orient_idx.item(), :, :
        ].unsqueeze(0)

        return TensorDict({
            "action": {
                "present_idx": present_idx,
                "present": present
            },
            "params": {
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
