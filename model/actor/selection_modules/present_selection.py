"""
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation.
"""
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
                # delete graph history before appending
                orient_features.append(features.detach())

            present_features = torch.stack(orient_features)
            all_features.append(present_features)

        self.all_present_features = torch.stack(all_features)

    # Returns with dims [BATCH] [ORIENTATION DIM]
    def _process_present_orients(self, grid_features, present_feat, p_idx):
        scores_list = []
        orient_td_list = []

        # If working on multiple samples, only modulated grid/score will differ
        for o_idx, orient_feat in enumerate(present_feat):
            # Use vmap on worker dim if exists
            if orient_feat.dim() > 2:
                score, modulated_grid = torch.vmap(
                    self.film
                )(grid_features, orient_feat)
                batch_dims = grid_features.shape[:2]
            else:
                score, modulated_grid = self.film(grid_features, orient_feat)
                batch_dims = (grid_features.shape[0],)

            # Add env batch dim first
            present_idx = torch.full(batch_dims, p_idx, dtype=torch.float32)
            orient_idx = torch.full(batch_dims, o_idx, dtype=torch.float32)

            # if batch dim is not a singleton, repeat it
            batched_orients = orient_feat.repeat(*batch_dims, 1)

            orient_td = TensorDict({
                "present_idx": present_idx,
                "orient_idx": orient_idx,
                "orient_features": batched_orients,
                "modulated_grid": modulated_grid
            }, batch_size=batch_dims)

            scores_list.append(score)
            orient_td_list.append(orient_td)

        scores_tensor = torch.stack(scores_list, dim=1)
        orient_tds = torch.stack(orient_td_list, dim=1)

        return scores_tensor, orient_tds

    def forward(self, tensordict):
        """ Gets scores for orientation / modulated grids for them """
        grid = tensordict.get("grid")
        present_count = tensordict.get("present_count")

        # Get features (handle worker dim if necessary)
        if grid.dim() > 4:
            grid_features = torch.vmap(self.grid_extractor)(grid)
        else:
            grid_features = self.grid_extractor(grid)

        # Orient predictions
        all_logits = []
        all_orient_tds = []

        # Mask out unavailable presents
        for p_idx, present_feat in enumerate(self.all_present_features):
            # Skip iteration if present is placed in all dims
            mask = present_count[..., p_idx] == 0

            if torch.all(mask):
                continue

            scores, orient_tds = self._process_present_orients(
                grid_features, present_feat, p_idx)

            # Cannot pick invalid presents
            scores[mask, :] = torch.tensor(-torch.inf, dtype=torch.float32)

            all_logits.append(scores)
            all_orient_tds.append(orient_tds)

        logits = torch.cat(
            all_logits,
            dim=1
        )
        orients = torch.cat(
            all_orient_tds,
            dim=1
        )

        return TensorDict({
            "orient_data": {
                "logits": logits,
                "orients": orients
            },
        })
