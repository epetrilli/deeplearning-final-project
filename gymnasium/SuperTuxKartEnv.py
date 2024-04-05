import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SuperTuxKartEnv(gym.Env):
    """
        Creates the SuperTuxKartEnvironment for agents to use.
        https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None):
        # TODO: Add custom boundaries for observation space

        self.goal_line_friendly = np.array([]) # TODO find a way to get these values from the sim
        self.goal_line_friendly = np.array([])

        self.action_space = spaces.Dict(
            {
                "acceleration": spaces.Box(0, 1, shape=(1,), dtype=np.float32), # 0 to 1 accel
                "steer": spaces.Box(-1, 1, shape=(1,), dtype=np.float32), # -1..1 left and right steering
                "brake": spaces.Discrete(2), # 0 not_breaking, 1 breaking
            }
        )
        self.observation_space = spaces.Dict( # 1 v 1 for now
            {
                "agent": spaces.Box(-1, 1, shape=(2,), dtype=np.float32), # xy coords of agent TODO: maybe use the actual XY coords
                "oppponent1": spaces.Box(-1, 1, shape=(2,), dtype=np.float32), # xy coords of oppenent
                "ball": spaces.Box(-1, 1, shape=(2,), dtype=np.float32) # xy coords of ball

            }
        )
        

    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def render(self, mode="human"):
        raise NotImplementedError
    
    def close(self):
        return super().close()
    
    def seed(self, seed=None):
        return

