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
        # worker dim (if exists) and batch dim
        batch_dims = choice_tensor.shape[:-1]
        orient_idxs = torch.argmax(choice_tensor, dim=len(batch_dims))

        orients = orient_td.get("orients")

        # if there are no batch dims (singleton)
        if len(batch_dims) == 0:
            orient = orients
        # if batch dim is more than 1
        elif batch_dims[-1] > 1:
            orient = orients.gather(len(batch_dims), orient_idxs)
        else:
            orient = orients[:, orient_idxs]

        present_idx = orient.get("present_idx")
        orient_idx = orient.get("orient_idx")
        orient_features = orient.get("orient_features")
        modulated_grid = orient.get("modulated_grid")

        if len(batch_dims) > 0:
            combined_features = torch.cat([
                orient_features.view(*batch_dims, -1),
                modulated_grid.view(*batch_dims, -1)
            ], dim=-1)
        else:
            combined_features = torch.cat([
                orient_features,
                modulated_grid
            ], dim=-1)

        x_mean = self.x_mean(combined_features)
        x_std = self.x_std(combined_features)
        y_mean = self.y_mean(combined_features)
        y_std = self.y_std(combined_features)

        # get present for partial action output
        if present_idx.numel() < 2:
            present = self.presents[  # type: ignore
                present_idx.item(), orient_idx.item(), :, :
            ].unsqueeze(0)
        else:
            present = self.presents[  # type: ignore
                present_idx.tolist(), orient_idx.tolist(), :, :
            ]

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
