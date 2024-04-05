import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class SuperTuxKartEnv(gym.Env):
    """
        Creates the SuperTuxKartEnvironment for agents to use.
        https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, 
                 render_mode="human", 
                 window_size=(255,255),
                 player_state=None,
                 opponent_state=None,
                 ):
        # TODO: Add custom boundaries for observation space
        self.window_size = window_size

        self.goal_line_friendly = np.array([]) # TODO find a way to get these values from the sim
        self.goal_line_oppponent = np.array([])

        self.action_space = spaces.Dict(
            {
                "acceleration": spaces.Box(0, 1, shape=(1,), dtype=np.float32), # 0..1 accel
                "steer": spaces.Box(-1, 1, shape=(1,), dtype=np.float32), # -1..1 left and right steering
                "brake": spaces.Discrete(2), # 0 not_breaking, 1 breaking
            }
        )
        self.observation_space = spaces.Dict( # 1 v 1 for now to get the environment started up
            {
                "agent": spaces.Box(-1, 1, shape=(2,), dtype=np.float32), # xy coords of agent TODO: maybe use the actual XY coords
                "oppponent1": spaces.Box(-1, 1, shape=(2,), dtype=np.float32), # xy coords of oppenent
                "ball": spaces.Box(-1, 1, shape=(2,), dtype=np.float32) # xy coords of ball
            }
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        

    def reset(self):
        raise NotImplementedError
    
    def step(self, action, player_state, opponent_state, soccer_state):
        # 1 do action

        # 2 get info from game to observation state
        
        
        raise NotImplementedError
    
    def render(self):
        return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # draw objects TODO: move to own method maybe
        pygame.draw.circle(
            canvas,
            (0,0,255), # blue
            self.observation_space['agent']
        )

        pygame.draw.circle(
            canvas,
            (255,0,0), # red
            self.observation_space['oppponent1']
        )

        pygame.draw.circle(
            canvas,
            (0,0,0),
            self.observation_space['ball']
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        raise NotImplementedError