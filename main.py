""" Main file"""

# import warnings

from model.ppo import PPO


def main():
    """ Main function """
    # block output shape warning as it doesn't seem to be important
    # warnings.filterwarnings("ignore", category=UserWarning)

    module = PPO()
    module.train()


if __name__ == "__main__":
    main()
