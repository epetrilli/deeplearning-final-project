import gymnasium as gym
from gymnasium import spaces

class SuperTuxKartEnv(gym.Env):
    """
        Creates the SuperTuxKartEnvironment for agents to use.
        https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict({}) # Add observations here
        self.action_space = spaces.Discrete() #Action space

    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def render(self, mode="human"):
        raise NotImplementedError
    
    def close(self):
        return super().close()

