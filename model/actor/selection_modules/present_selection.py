"""
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation.
"""
from itertools import chain
from tensordict import TensorDict
from torch import nn
import torch

from model.actor.feature_modules.film import FiLM
from model.actor.feature_modules.grid_features import GridExtractor
from model.actor.feature_modules.present_features import PresentExtractor


class PresentSelectionActor(nn.Module):
    """ Policy nn for PresentEnv to choose which present to place. """

    def __init__(self, presents: torch.Tensor, device: torch.device):
        super().__init__()

        self.device = device

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
                # add batch, channel dim
                orient = orient.unsqueeze(0).unsqueeze(0)
                features = self.present_extractor(orient)
                orient_features.append(features)

            present_features = torch.stack(orient_features)
            all_features.append(present_features)

        self.all_present_features = torch.stack(all_features)

    def _process_present_orients(self, grid_features, present_feat, p_idx):
        scores = []
        orient_tds = []

        for o_idx, orient_feat in enumerate(present_feat):
            score, modulated_grid = self.film(grid_features, orient_feat)

            batch_dim = score.shape[0]

            # Getting idx tensors of correct batch size
            present_idx = torch.tensor(
                p_idx, dtype=torch.uint8, device=self.device).unsqueeze(0).repeat(batch_dim)
            orient_idx = torch.tensor(
                o_idx, dtype=torch.uint8).unsqueeze(0).repeat(batch_dim)

            if batch_dim > 1:
                batched_orients = orient_feat.repeat(batch_dim, 1, 1)
            else:
                batched_orients = orient_feat

            orient_td = TensorDict({
                "present_idx": present_idx,
                "orient_idx": orient_idx,
                "orient_features": batched_orients,
                "modulated_grid": modulated_grid
            }, batch_size=batch_dim)

            scores.append(score)
            orient_tds.append(orient_td)

        return scores, orient_tds

    def forward(self, tensordict):
        """ Gets scores for orientation / modulated grids for them """
        grid = tensordict.get("grid")
        present_count = tensordict.get("present_count")
        # Get features
        grid_features = self.grid_extractor(grid)

        # orient predictions
        present_tuples = []

        for p_idx, present_feat in enumerate(self.all_present_features):
            if present_count.dim() > 2:
                mask = present_count[:, :, p_idx] != 0
            else:
                mask = present_count[:, p_idx] != 0
            present_tuple = self._process_present_orients(
                grid_features[mask], present_feat, p_idx)
            present_tuples.append(present_tuple)

        # transpose tuples
        orient_logits, orient_tds = zip(
            *present_tuples)

        logits = torch.stack(
            list(
                chain.from_iterable(orient_logits)
            ),
            dim=1
        ).squeeze(-1)
        orients = torch.stack(
            list(
                chain.from_iterable(orient_tds)
            ),
            dim=1
        )

        return TensorDict({
            "orient_data": {
                "logits": logits,
                "orients": orients
            },
        })
