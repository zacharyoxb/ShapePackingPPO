""" Main file"""


from model.ppo import PPO


def main():
    """ Main function """
    module = PPO()
    module.train()


if __name__ == "__main__":
    main()
