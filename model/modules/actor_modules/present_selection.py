""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from torch import nn
import torch


class PresentSelectionActor(nn.Module):
    """ Policy nn for PresentEnv to choose which present to place. """

    def __init__(self, device=torch.device("cpu")):
        super().__init__()

    def forward(self, tensordict):
        """ Gets scores for orientation of each present to choose which to place """
        # Get only presents still in "play"
        presents = tensordict.get("presents")
