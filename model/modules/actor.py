""" Actor Policy implementation for use with my present packing environment. """
from dataclasses import dataclass
from tensordict import TensorDict
from torch import nn
import torch

from model.modules.feature_extractor import FeatureExtractor


@dataclass
class Heads:
    """ Holds all model heads """
    present_idx: nn.Sequential
    x_loc: nn.Sequential
    x_scale: nn.Sequential
    y_loc: nn.Sequential
    y_scale: nn.Sequential
    rot: nn.Sequential
    flip: nn.Sequential


class PresentActor(nn.Module):
    """ Policy nn for PresentEnv with spatial awareness """

    def __init__(self, device=torch.device("cpu")):
        super().__init__()

        self.flatten = nn.Flatten()
        self.extractor = FeatureExtractor()
        self.device = device

        # Output distributions
        present_idx = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 presents
        ).to(self.device)
        x_loc = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        x_scale = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        y_loc = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        y_scale = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        rot = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 rotations
        ).to(self.device)
        flip = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 binary decisions
        ).to(self.device)

        self.heads = Heads(present_idx, x_loc, x_scale,
                           y_loc, y_scale, rot, flip)

    def forward(self, tensordict):
        """ Forward function for running of nn """
        # get present count for masking out impossible choices
        present_count = tensordict.get(
            "present_count").unsqueeze(0)

        # get features
        all_features = self.extractor(tensordict)

        # calculate logits
        present_idx_logits = self.heads.present_idx(all_features)
        rot_logits = self.heads.rot(all_features)
        flip_logits = self.heads.flip(all_features)

        x_loc = self.heads.x_loc(all_features)
        x_scale = torch.exp(self.heads.x_scale(all_features))
        y_loc = self.heads.y_loc(all_features)
        y_scale = torch.exp(self.heads.y_scale(all_features))

        # mask out unavailable presents from logits
        idx_mask = (present_count > 0).float()
        present_idx_logits = present_idx_logits + idx_mask.log()

        return TensorDict({
            "params": {
                "present_idx": {
                    "logits": present_idx_logits
                },
                "rot": {
                    "logits": rot_logits
                },
                "flip": {
                    "logits": flip_logits
                },
                "x": {
                    "loc": x_loc,
                    "scale": x_scale
                },
                "y": {
                    "loc": y_loc,
                    "scale": y_scale},
            },
        }, batch_size=torch.Size([]), device=self.device)
