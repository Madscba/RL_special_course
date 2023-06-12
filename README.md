Introduction
--------
This repository hold implementations of the following reinforcement learning methods: Double Deep Q-Learning (DDQN), Reinforce, Actor-Critic (AC), Soft Actor-Critic (SAC, first version, https://arxiv.org/abs/1801.01290).
Below is instructions for how to set up a virtual environment that can run the code accompanied by instructions on how to configure each method followed with results from the training. In the end a description of the codebase structure can be found. The methods have been tuned to play the GymAI environment LunarLander either with discrete or continuous action space.
The DDQN method only function with the discrete action space (Lunarlander-v2 environment), and SAC only in the continuous action space (LunarlanderContinuous-v2 environment). Both Reinforce and AC work in both. For each method a grid search of different parameters has been done, and once a set of parameters was found to work reasonably no further extensive search was performed. This means that the methods might not be directly comparably as the best hyperparameters are not guaranteed to be similar to the ones I have used and reported.

Instructions on how to run the code
---
Setting up the environment
-
Python version 3.8.6 was used for this project.

A typical requirement when the ´Gymnasium´ environments such as LunarLander are used is the module [Swig](www.swig.org/download.html), which need to be downloaded and setup according to their guide.
```
git clone 
conda env create --name rl_venv --file environment.yml 
```
main.py
-
To train an agent the main.py is used. Hyperparameters are set in the file utils/argparser.py or as a flag if main.py is run from the terminal. For each method a command for running the algorithm is provided. You can train a particular agent using these commands and results showing the training progress will be visualized. For convenience, the results using these commands are also included in /results/final/<method>. For each method 3 graphs are included. The first plot "Reward Progres" with steps as the x-axis show the average reward pr. step. The average is taken over non-overlapping sequences of 10.000 steps. The second plot shows length (frames/steps) for each episode that the agents has played. The last plot shows the total reward per episode the agents has received for each played episode. Training each episode takes between 20 minutes and 5 hours using a CPU (AMD Ryzen 9 5900X 12-Core Processor, 3701 Mhz, 12 Core(s), 24 Logical Processor(s))



DDQN:
-------------

python main.py --env_name "LunarLander-v2" --visualize 0 --n_steps 400000 --frame_interval 10000 --n_env 1 --algorithm "DDQN" --gamma 0.99 --hidden_size 64 --lr 0.001 --eps 1 --eps_decay 0.001 --min_eps 0.05 --batch_size 32 --use_replay 1


![Image](/hand_in/results/final/DDQN/avg_frame_rewards.png)

![Image](/hand_in/results/final/DDQN/episode_lengths.png)

![Image](/hand_in/results/final/DDQN/episode_rewards.png)


REINFORCE (discrete):
-------------

python main.py --env_name "LunarLander-v2" --visualize 0 --n_steps 400000 --frame_interval 10000 --n_env 1 --algorithm "REINFORCE" --gamma 0.99 --hidden_size 256 --lr 0.001 --eps 1 --eps_decay 0.0001 --min_eps 0.05 --use_replay 0



![Image](/hand_in/results/final/REINFORCE/discrete/avg_frame_rewards.png)

![Image](/hand_in/results/final/REINFORCE/discrete/episode_lengths.png)

![Image](/hand_in/results/final/REINFORCE/discrete/episode_rewards.png)

REINFORCE (continuous):
-------------

python main.py --env_name "LunarLanderContinuous-v2" --visualize 0 --n_steps 400000 --frame_interval 10000 --n_env 1 --algorithm "REINFORCE" --gamma 0.99 --hidden_size 256 --lr 0.001 --eps 1 --eps_decay 0.0001 --min_eps 0.05 --use_replay 0



![Image](/hand_in/results/final/REINFORCE/continuous/avg_frame_rewards.png)

![Image](/hand_in/results/final/REINFORCE/continuous/episode_lengths.png)

![Image](/hand_in/results/final/REINFORCE/continuous/episode_rewards.png)

Actor-Critic (continuous)
-------------

python main.py --env_name "LunarLanderContinuous-v2" --visualize 0 --n_steps 600000 --frame_interval 10000 --n_environments 1 --algorithm "AC" --gamma 0.99 --hidden_size 512 --lr 0.0001 --eps 1 --eps_decay 0.001 --min_eps 0.05 --grad_clipping 0


![Image](/hand_in/results/final/AC/continuous/avg_frame_rewards.png)

![Image](/hand_in/results/final/AC/continuous/episode_lengths.png)

![Image](/hand_in/results/final/AC/continuous/episode_rewards.png)

Actor-Critic (discrete)
-------------

python main.py --env_name "LunarLanderContinuous-v2" --visualize 0 --n_steps 600000 --frame_interval 10000 --n_environments 1 --algorithm "AC" --gamma 0.99 --hidden_size 512 --lr 0.0001 --eps 1 --eps_decay 0.001 --min_eps 0.05 --grad_clipping 0

![Image](/hand_in/results/final/AC/discrete/avg_frame_rewards.png)

![Image](/hand_in/results/final/AC/discrete/episode_lengths.png)

![Image](/hand_in/results/final/AC/discrete/episode_rewards.png)

SAC:
-------------
python main.py --seed 3 --env_name LunarLanderContinuous-v2 --log_format "default" --frame_interval 10000 --visualize 0 --n_steps 600000 --n_env 1 --algorithm SAC_v0 --gamma 0.99 --hidden_size 256 --lr 0.0003 --batch_size 256 --grad_clipping 0 --tau 0.005 --reward_scale 2.0


![Image](/hand_in/results/final/SAC/avg_frame_rewards.png)

![Image](/hand_in/results/final/SAC/episode_lengths.png)

![Image](/hand_in/results/final/SAC/episode_rewards.png)



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


'results'
-
This folder contains results achieved with the different methods. For methods both working with discrete a continuous there are subfolders for each domain.