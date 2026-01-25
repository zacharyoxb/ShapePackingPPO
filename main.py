""" Main file"""


from data.reader import get_data
from model.ppo import PPO


def main():
    """ Main function """
    data_buffer = get_data("testinput.txt")

    module = PPO(data_buffer)

    module.train()


if __name__ == "__main__":
    main()
