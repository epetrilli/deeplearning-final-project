Important Links and Weekly Objectives/Meetings

## Pdf Links: 
  - http://cs231n.stanford.edu/reports/2017/pdfs/624.pdf
  - https://www.researchgate.net/publication/357897741_Deep_Learning_Project_of_a_SuperTuxKart_ice-hockey_player
  - https://arxiv.org/pdf/1702.05663.pdf
  - https://arxiv.org/abs/1707.06347
  - https://gymnasium.farama.org/index.html


## Weekly Objectives

### Meeting 0
  - ~~Create git repo? - Thads~~
  - ~~Create group in canvas - Seb~~ 
  - ~~Send github usernames to Thads - Everyone~~
      SMozaffar
      Epetrill
      sebastien-savard


### Meeting 1
  - Brainstorming 
  - Research on state-based agents
  - Run a test-agent
  - Read and understand structure of starter code


### Meeting 2
  - Gymnasium: Library for ML environments
  - PPO and/or DPO algorithms
  - Prototype code in 'player.py'
  - Research on reinforcement learning
  - Use a 'discounting factor' for uncertain future events (Bellman Optimality Equation)
  - Sebastien ran test agent, Thads is working on running some (for issues related to running agents, look at EdDiscussion for fixes)
  - DECIDED: Start with state-based approach, with potential for shifting to image-based approach based on unanimous deision after a certain deadline
  - Start to think about going into the next meeting with directed tasks and goals -> Wednesday: Identify and designate tasks
  - Figure out version control
  


#### I propose we model our approach on https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
#### https://www.youtube.com/watch?v=hlv79rcHws0
#### We must implement our environment (gym) and then can kind of do similar implementation as above, with the Team class being similar to the Agent class in the example 
#### See proposed team in state_agent/player.py



# [Draft] Final Project Tasks

## 1. Model the Ice Hockey PySuperTux Kart game as a gymnasium evnironment

- **Task 1**: Model the observation state so that the agent can accurately perceive the game environment and make informed decisions. (can look at example state based agents)
- **Task 2**: Define the action space (some discrete, some continuous items)


## 2. Create a Reward Function

Create a reward function to guide the agent's learning process towards successful strategies. Is this a part of a gymnasium environment? 

- **Task 1**: Identify key behaviors to reward (e.g., scoring goals, blocking shots, maintaining possession).
- **Task 2**: Design a formula for calculating the reward, considering immediate and long-term rewards.
- **Task 3**: Implement the reward function, ensuring it can dynamically respond to the agent's actions and the game state.


## 3. Implement the Team class

- **Task 1**: Implement Actor network (outputs a probability distribution over actions given the current state of the environment) 
- **Task 2**: Implement Critic network (estimates the value of being in a given state, used along with rewards to compute advantage) (should be similar structure to Actor network)
- **Task 3**: Implement trajectory memory object, store trajectories from simulations for training
- **Task 4**: Implement team class (hyper parameters here?)
- **Task 5**: Implement load_models() and save_models() utils for inference and training. 
- **Task 6**: Implement new_match (does kart selection matter?)
- **Task 7**: Implement choose_action (this calls Actor network)
- **Task 8**: Implement act() function, main method called each time step. wrapper around choose_action to wrap as pystk action
- **Task 9**: Implement learn() function, this is where bulk of PPO's learning logic goes

## 4. Implement the Training Environment

- **Task 1**: Set up a script/environemnt w/ training loop where the Team can play games, we will call learn() to train. Similar to main.py in example

## 5. Begin training the agent using PPO 

Use the PPO algorithm to train the RL agent, optimizing its policy for winning gameplay.

### Subtasks:

- **Task 1**: Select and configure the PPO algorithm parameters / other hyper params
- **Task 2**: Evaluate and iterate
