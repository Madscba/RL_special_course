Introduction
--------
This repository hold implementations of the following reinforcement learning methods: Double Deep Q-Learning (DDQN), Reinforce, Actor-Critic (AC), Soft Actor-Critic (SAC, first version, https://arxiv.org/abs/1801.01290).
Below is an explanation of the structure of the code base, pre-requisites and instructions for running the code.
The methods are tailored to either discrete (DDQN & Reinforce) or continuous (AC, SAC) action space and have been tuned.



Codebase Structure
------
Here's an overview of the folders and the corresponding classes/modules:

'agents/'
-
This folder contains classes representing different RL agents. Each agent is in a separate file. THe most important functionality for each agent is the initialization of a policy, and two methods. One to follow the policy given an observation (agent.follow_policy()) and secondly learn from experience (update_policy()).

base_agent.py: An abstract base class for RL agents with common functionality.


'environments/'
-
This folder holds a script to import the gym-environments used in this project.

'models/'
-
This folder contains neural network classes representing policies for the agents. The following files can be included:

base_model.py: An abstract base class for models with common functionality.


'utils/'
-
This folder contains utility modules such as an argparser, replay_buffer, logger evaluation script. These are used and shared by the RL agents.


main.py
-
To train an agent the main.py is used. Using the configuration in the utils/argparser.py. Running the code using the terminal with the following arguments, you can train a particular agent and results showing the training progress will be visualized.
For convenience, the results are also included in /results/final/<method>



'results'
-
This folder contains results achieved with the different methods.

DDQN:
-------------

python main.py --env_name "LunarLander-v2" --visualize 0 --n_steps 400000 --frame_interval 10000 --n_env 1 --algorithm "DDQN" --gamma 0.99 --hidden_size 64 --lr 0.001 --eps 1 --eps_decay 0.001 --min_eps 0.05 --batch_size 32 --use_replay 1


![Image](/hand_in/results/final/DDQN/avg_step_episode.png)

![Image](/hand_in/results/final/DDQN/length_episode.png)

![Image](/hand_in/results/final/DDQN/rew_episode.png)


REINFORCE:
-------------

python main.py --env_name "LunarLander-v2" --visualize 0 --n_steps 400000 --frame_interval 10000 --n_env 1 --algorithm "REINFORCE" --gamma 0.99 --hidden_size 256 --lr 0.001 --eps 1 --eps_decay 0.0001 --min_eps 0.05 --use_replay 0



![Image](/hand_in/results/final/REINFORCE/avg_step_episode.png)

![Image](/hand_in/results/final/REINFORCE/length_episode.png)

![Image](/hand_in/results/final/REINFORCE/rew_episode.png)

Actor-Critic
-------------

python main.py --env_name "LunarLanderContinuous-v2" --visualize 0 --n_steps 600000 --frame_interval 10000 --n_environments 1 --algorithm "AC" --gamma 0.99 --hidden_size 512 --lr 0.0001 --eps 1 --eps_decay 0.001 --min_eps 0.05 --grad_clipping 0


![Image](/hand_in/results/final/AC/avg_step_episode.png)

![Image](/hand_in/results/final/AC/length_episode.png)

![Image](/hand_in/results/final/AC/rew_episode.png)

SAC:
-------------

python main.py --env_name "LunarLanderContinuous-v2" --visualize False --n_steps 200000 --frame_interval 10000 --n_environments 1 --algorithm "REINFORCE" --gamma 0.95 --hidden_size 32 --lr 0.001 --eps 1 --eps_decay 0.0001 --min_eps 0.05


![Image](/hand_in/results/final/SAC/avg_step_episode.png)

![Image](/hand_in/results/final/SAC/length_episode.png)

![Image](/hand_in/results/final/SAC/rew_episode.png)