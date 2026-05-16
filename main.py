""" Main file """

import warnings
import torch

from model.ppo import PPO


def main():
    """ Main function """
    warnings.filterwarnings("error", category=UserWarning)
    torch.autograd.set_detect_anomaly(True)

    module = PPO()
    module.train()


if __name__ == "__main__":
    main()
