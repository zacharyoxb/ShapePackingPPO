""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from tensordict import TensorDict
from torch import nn
import torch

from model.actor.selection_modules.td_utils import OrientationEntry
from model.actor.feature_modules.film import FiLM
from model.actor.feature_modules.grid_features import GridExtractor
from model.actor.feature_modules.present_features import PresentExtractor


class PresentSelectionActor(nn.Module):
    """ Policy nn for PresentEnv to choose which present to place. """

    def __init__(self, presents: torch.Tensor, device: torch.device):
        super().__init__()

        # Presents shouldn't be learnable/modifiable nor in state_dict
        self.register_buffer("all_present_features",
                             torch.tensor([]), persistent=False)

        self.grid_extractor = GridExtractor(device)
        self.present_extractor = PresentExtractor(device)
        self.film = FiLM(device)

        self._feature_init(presents)

    def _feature_init(self, presents):
        all_features = []

        for present in presents:
            orient_features = []
            for orient in present:
                # add channel dim
                present = present.unsqueeze(0)

                features = self.present_extractor(orient)
                orient_features.append(features)

            present_features = torch.stack(orient_features)
            all_features.append(present_features)

        self._all_present_features = torch.stack(all_features)

    def forward(self, tensordict):
        """ Gets scores for orientation / modulated grids for them """
        grid = tensordict.get("grid")
        present_count = tensordict.get("present_count")

        # Extract features
        grid_features = self.grid_extractor(grid)

        orient_logits = torch.tensor([])
        orient_data = []

        # Calculate scores for each orientation
        for present_idx, present_features in enumerate(self._all_present_features):
            if present_count[present_idx] == 0:
                continue

            for orient_idx, orient_features in enumerate(present_features):
                score, modulated_grid = self.film(
                    grid_features, orient_features)

                orient_logits.add(score)
                orient_data.append(
                    OrientationEntry(
                        torch.tensor(present_idx),
                        torch.tensor(orient_idx),
                        self._all_present_features[present_idx][orient_idx],
                        modulated_grid
                    )
                )

        choice_idx = torch.multinomial(orient_logits, len(orient_data))

        return TensorDict({
            "present_data": TensorDict.from_dataclass(orient_data[choice_idx])
        })
