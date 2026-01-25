""" Actor Policy implementation for use with my present packing environment. """
from dataclasses import dataclass
from tensordict import TensorDict
from torch import nn

from model.modules.feature_extractor import FeatureExtractor


@dataclass
class Heads:
    """ Holds all model heads """
    present_idx_head: nn.Sequential
    x_head: nn.Sequential
    y_head: nn.Sequential
    rot_head: nn.Sequential
    flip_head: nn.Sequential


class PresentActor(nn.Module):
    """ Policy nn for PresentEnv with spatial awareness """

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.extractor = FeatureExtractor()

        # Output distributions
        present_idx_head = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 presents
        )
        rot_head = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 rotations
        )
        flip_head = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 binary decisions
        )

        # Softmax transform will be used for these continuous estimates
        x_head = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        y_head = nn.Sequential(
            nn.Linear(self.extractor.combined_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.heads = Heads(present_idx_head, x_head,
                           y_head, rot_head, flip_head)

    def forward(self, tensordict):
        """ Forward function for running of nn """
        # get present count for masking out impossible choices
        present_count = tensordict.get("present_count").unsqueeze(0)

        # get features
        all_features = self.extractor(tensordict)

        # calculate logits
        present_idx_logits = self.heads.present_idx_head(all_features)
        rot_logits = self.heads.rot_head(all_features)
        flip_logits = self.heads.flip_head(all_features)

        x_gauss = self.heads.x_head(all_features)
        y_gauss = self.heads.y_head(all_features)

        # mask out unavailable presents from logits
        idx_mask = (present_count > 0).float()
        present_idx_logits = present_idx_logits + idx_mask.log()

        return TensorDict({
            "present_idx_logits": present_idx_logits,
            "rot_logits": rot_logits,
            "flip_logits": flip_logits,
            "x": x_gauss,
            "y": y_gauss,
        }, batch_size=present_idx_logits.shape[0])
