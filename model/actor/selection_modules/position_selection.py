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
        batch_dims = orient_td.batch_size

        orients = orient_td.get("orients")

        orient_idxs = torch.argmax(
            choice_tensor, dim=-1)

        if batch_dims:
            batch_idx = torch.meshgrid(
                *[torch.arange(d) for d in batch_dims], indexing='ij')
            orient = orients[(*batch_idx, orient_idxs)]
        else:
            orient = orients[..., orient_idxs].squeeze(1)

        present_idx = orient.get("present_idx")
        orient_idx = orient.get("orient_idx")
        orient_features = orient.get("orient_features")
        modulated_grid = orient.get("modulated_grid")

        combined_features = torch.cat([
            orient_features,
            modulated_grid
        ], dim=-1)

        if len(batch_dims) > 1:
            x_mean = torch.vmap(self.x_mean)(combined_features).squeeze(-1)
            x_std = torch.vmap(self.x_std)(combined_features).squeeze(-1)
            y_mean = torch.vmap(self.y_mean)(combined_features).squeeze(-1)
            y_std = torch.vmap(self.y_std)(combined_features).squeeze(-1)
        else:
            x_mean = self.x_mean(combined_features).squeeze(-1)
            x_std = self.x_std(combined_features).squeeze(-1)
            y_mean = self.y_mean(combined_features).squeeze(-1)
            y_std = self.y_std(combined_features).squeeze(-1)

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
