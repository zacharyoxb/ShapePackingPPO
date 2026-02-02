""" Main file"""

import warnings

from model.ppo import PPO


def main():
    """ Main function """
    warnings.filterwarnings("error", category=UserWarning)

    module = PPO()
    module.train()


if __name__ == "__main__":
    main()
