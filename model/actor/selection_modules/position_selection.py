""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from torch import nn


class PresentPositionActor(nn.Module):
    """ Policy nn for PresentEnv to choose where to place present. """

    def __init__(self, device):
        super().__init__()

        self.device = device

        # Normal coordinate gen networks go here

    def forward(self, tensordict):
        """ Choose a placement position for the selected present in present_data """
