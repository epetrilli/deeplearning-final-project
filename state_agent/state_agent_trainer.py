import gymnasium as gym
from .utils import init_device

def train(args):
    from os import path

    # model = Model() # TO DO
    device = init_device()
    # model.to(device)

    env = gym.make('SuperTuxCartIceHockey-v1')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')

    args = parser.parse_args()
    train(args)