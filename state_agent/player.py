
class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players

        # call reset() on gym env to init the env state

        return ['tux'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        return [dict(acceleration=1, steer=0)] * self.num_players


class ProposedTeam:
    agent_type = 'state'
    
    def __init__(self):
        """
        Initialize the team agent.
        """
        self.team = None
        self.num_players = None
        #self.actor = ActorNetwork(...) 
        #self.critic = CriticNetwork(...)  
        #self.memory = Memory(...) 
        
        # Hyperparameters (fake values)
        self.gamma = 0.99 
        self.alpha = 0.0003
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.batch_size = 64
        self.n_epochs = 10  # Number of epochs to train for each learning step
    
    def load_models(self):
        """
        Load the model parameters for both the actor and critic networks.
        """
        pass
    
    def save_models(self):
        """
        Save the model parameters for both the actor and critic networks.
        """
        pass
    
    def new_match(self, team: int, num_players: int) -> list:
        """
        Prepares the team for a new match.
        """
        pass
    
    def choose_action(self, player_state, opponent_state, soccer_state):
        """
        Decide actions for each player based on the current game state.
        """
        # This will involve processing the states through the actor network
        # and deciding on actions
        
        pass
    
    def learn(self):
        """
        Update the actor and critic networks based on stored experiences.
        """
        # This is where the bulk of PPO's learning logic will go.
        
        # For each epoch:
            # Generate batches from memory
            # Calculate advantages
            # Update networks based on PPO's objective function
            # Clear memory after updating
            
        # Placeholder for learning logic
        pass
    
    def act(self, player_state, opponent_state, soccer_state):
        """
        Main method called each timestep. Wrapper around choose_action.
        """
        actions = self.choose_action(player_state, opponent_state, soccer_state)
        # Convert actions to the required format
        # TODO action formatting
        return actions