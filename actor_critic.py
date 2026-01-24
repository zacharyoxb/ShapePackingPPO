""" Policy implementation for use with my present packing environment. """
from dataclasses import dataclass
from tensordict import TensorDict
import torch
from torch import nn


@dataclass
class FeatureExtractor:
    """ Holds all feature extractors """
    grid_encoder: nn.Sequential
    present_encoder: nn.Sequential
    present_count_encoder: nn.Sequential


@dataclass
class Heads:
    """ Holds all model heads """
    present_idx_head: nn.Sequential
    x_head: nn.Sequential
    y_head: nn.Sequential
    rot_head: nn.Sequential
    flip_head: nn.Sequential
    value_head: nn.Sequential


class PresentActorCritic(nn.Module):
    """ Policy nn for PresentEnv with spatial awareness """

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # Process grid
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

        # Process present
        self.present_encoder = nn.Sequential(
            nn.Linear(6 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Process present_count
        self.present_count_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        combined_features = 128 + 128 + 32

        # Output distributions
        present_idx_head = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 presents
        )
        rot_head = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 rotations
        )
        flip_head = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 binary decisions
        )

        # Softmax transform will be used for these continuous estimates
        x_head = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        y_head = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Critic
        value_head = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # single value estimate
        )

        self.heads = Heads(present_idx_head, x_head,
                           y_head, rot_head, flip_head, value_head)

    def forward(self, tensordict):
        """ Forward function for running of nn """
        # get observations with batch dimensions
        grid = tensordict.get("grid").unsqueeze(0).unsqueeze(0)
        presents = tensordict.get("presents").unsqueeze(0)
        present_count = tensordict.get("present_count").unsqueeze(0)

        # Concat all features
        all_features = torch.cat([
            self.grid_encoder(grid),
            self.present_encoder(
                presents.flatten(start_dim=1)),
            self.present_count_encoder(
                present_count)
        ], dim=1)

        # calculate logits
        present_idx_logits = self.heads.present_idx_head(all_features)
        rot_logits = self.heads.rot_head(all_features)
        flip_logits = self.heads.flip_head(all_features)

        x_gauss = self.heads.x_head(all_features)
        y_gauss = self.heads.y_head(all_features)

        # mask out unavailable presents from logits
        idx_mask = (present_count > 0).float()
        present_idx_logits = present_idx_logits + idx_mask.log()

        # get value
        value = self.heads.value_head(all_features)

        return TensorDict({
            "present_idx_logits": present_idx_logits,
            "rot_logits": rot_logits,
            "flip_logits": flip_logits,
            "x": x_gauss,
            "y": y_gauss,
            "value": value
        }, batch_size=present_idx_logits.shape[0])
