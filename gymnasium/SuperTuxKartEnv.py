import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import logging

METADATA = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
XY_COORDINATE_LOWER_BOUND = -1000
XY_COORDINATE_UPPER_BOUND = 1000
TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000

class SuperTuxKartEnv(gym.Env):
    """
        Creates the SuperTuxKartEnvironment for agents to use.
        https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
    """
    def __init__(self, 
                 window_size=(255,255),
                 use_graphics=False,
                 logging_level=None):
        # TODO: Add custom boundaries for observation space
        import pystk
        self._pystk = pystk
        if logging_level is not None:
            logging.basicConfig(level=logging_level)

        # Fire up pystk
        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)
        self.window_size = window_size
        self.goal_line_friendly = np.array([]) # TODO find a way to get these values from the sim
        self.goal_line_opponent = np.array([])
        self.action_space = spaces.Dict(
            {
                "acceleration": spaces.Box(0, 1, shape=(1,), dtype=np.float32), # 0..1 accel
                "steer": spaces.Box(-1, 1, shape=(1,), dtype=np.float32), # -1..1 left and right steering
                "brake": spaces.Discrete(2), # 0 not_breaking, 1 breaking
            }
        )
        self.observation_space = spaces.Dict( # 1 v 1 for now to get the environment started up
            {
                "agent": spaces.Box(XY_COORDINATE_LOWER_BOUND, XY_COORDINATE_UPPER_BOUND, shape=(2,), dtype=np.float32), # xy coords of agent
                "opponent1": spaces.Box(XY_COORDINATE_LOWER_BOUND, XY_COORDINATE_UPPER_BOUND, shape=(2,), dtype=np.float32), # xy coords of opponent
                "ball": spaces.Box(XY_COORDINATE_LOWER_BOUND, XY_COORDINATE_UPPER_BOUND, 1, shape=(2,), dtype=np.float32) # xy coords of ball
            }
        )

        self.window = None
        self.clock = None

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)
        

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

        # draw objects
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
        if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
            self._pystk.clean()
    
    def seed(self, seed=None):
        raise NotImplementedError
    

class Match:
    """
        Do not create more than one match per process (use ray to create more)
    """
    def __init__(self, use_graphics=False, logging_level=None):
        # DO this here so things work out with ray
        import pystk
        self._pystk = pystk
        if logging_level is not None:
            logging.basicConfig(level=logging_level)

        # Fire up pystk
        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)

    def __del__(self):
        if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
            self._pystk.clean()

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    @classmethod
    def _r(cls, f):
        if hasattr(f, 'remote'):
            return f.remote
        if hasattr(f, '__call__'):
            if hasattr(f.__call__, 'remote'):
                return f.__call__.remote
        return f

    @staticmethod
    def _g(f):
        from .remote import ray
        if ray is not None and isinstance(f, (ray.types.ObjectRef, ray._raylet.ObjectRef)):
            return ray.get(f)
        return f

    def _check(self, team1, team2, where, n_iter, timeout):
        _, error, t1 = self._g(self._r(team1.info)())
        if error:
            raise MatchException([0, 3], 'other team crashed', 'crash during {}: {}'.format(where, error))

        _, error, t2 = self._g(self._r(team2.info)())
        if error:
            raise MatchException([3, 0], 'crash during {}: {}'.format(where, error), 'other team crashed')

        logging.debug('timeout {} <? {} {}'.format(timeout, t1, t2))
        return t1 < timeout, t2 < timeout

    def run(self, team1, team2, num_player=1, max_frames=MAX_FRAMES, max_score=3, record_fn=None, timeout=1e10,
            initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], verbose=False):
        RaceConfig = self._pystk.RaceConfig

        logging.info('Creating teams')

        # Start a new match
        t1_cars = self._g(self._r(team1.new_match)(0, num_player)) or ['tux']
        t2_cars = self._g(self._r(team2.new_match)(1, num_player)) or ['tux']

        t1_type, *_ = self._g(self._r(team1.info)())
        t2_type, *_ = self._g(self._r(team2.info)())

        if t1_type == 'image' or t2_type == 'image':
            assert self._use_graphics, 'Need to use_graphics for image agents.'

        # Deal with crashes
        t1_can_act, t2_can_act = self._check(team1, team2, 'new_match', 0, timeout)

        # Setup the race config
        logging.info('Setting up race')

        race_config = RaceConfig(track=TRACK_NAME, mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * num_player)
        race_config.players.pop()
        for i in range(num_player):
            race_config.players.append(self._make_config(0, hasattr(team1, 'is_ai') and team1.is_ai, t1_cars[i % len(t1_cars)]))
            race_config.players.append(self._make_config(1, hasattr(team2, 'is_ai') and team2.is_ai, t2_cars[i % len(t2_cars)]))

        # Start the match
        logging.info('Starting race')
        race = self._pystk.Race(race_config)
        race.start()
        race.step()

        state = self._pystk.WorldState()
        state.update()
        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))

        for it in range(max_frames):
            logging.debug('iteration {} / {}'.format(it, MAX_FRAMES))
            state.update()

            # Get the state
            team1_state = [to_native(p) for p in state.players[0::2]]
            team2_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            team1_images = team2_images = None
            if self._use_graphics:
                team1_images = [np.array(race.render_data[i].image) for i in range(0, len(race.render_data), 2)]
                team2_images = [np.array(race.render_data[i].image) for i in range(1, len(race.render_data), 2)]

            # Have each team produce actions (in parallel)
            if t1_can_act:
                if t1_type == 'image':
                    team1_actions_delayed = self._r(team1.act)(team1_state, team1_images)
                else:
                    team1_actions_delayed = self._r(team1.act)(team1_state, team2_state, soccer_state)

            if t2_can_act:
                if t2_type == 'image':
                    team2_actions_delayed = self._r(team2.act)(team2_state, team2_images)
                else:
                    team2_actions_delayed = self._r(team2.act)(team2_state, team1_state, soccer_state)

            # Wait for the actions to finish
            team1_actions = self._g(team1_actions_delayed) if t1_can_act else None
            team2_actions = self._g(team2_actions_delayed) if t2_can_act else None

            new_t1_can_act, new_t2_can_act = self._check(team1, team2, 'act', it, timeout)
            if not new_t1_can_act and t1_can_act and verbose:
                print('Team 1 timed out')
            if not new_t2_can_act and t2_can_act and verbose:
                print('Team 2 timed out')

            t1_can_act, t2_can_act = new_t1_can_act, new_t2_can_act

            # Assemble the actions
            actions = []
            for i in range(num_player):
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
                actions.append(a1)
                actions.append(a2)

            if record_fn:
                self._r(record_fn)(team1_state, team2_state, soccer_state=soccer_state, actions=actions,
                                   team1_images=team1_images, team2_images=team2_images)

            logging.debug('  race.step  [score = {}]'.format(state.soccer.score))
            if (not race.step([self._pystk.Action(**a) for a in actions]) and num_player) or sum(state.soccer.score) >= max_score:
                break

        race.stop()
        del race

        return state.soccer.score

    def wait(self, x):
        return x

class MatchException(Exception):
    def __init__(self, score, msg1, msg2):
        self.score, self.msg1, self.msg2 = score, msg1, msg2

def to_native(o):
    # Super obnoxious way to hide pystk
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)