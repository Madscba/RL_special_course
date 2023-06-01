# A class wrapping an OpenAI Gym environment. Vectorized environments are possible as well.
import gymnasium as gym
import numpy as np


def get_envs_vec(env_id: str = "LunarLander-v2", num_envs: int = 10):
    envs = gym.vector.make(id=env_id, num_envs=num_envs)
    starting_state, _ = envs.reset()
    return envs, starting_state

def get_envs(env_id: str = "LunarLander-v2", num_envs: int = 10):
    envs = gym.make(id=env_id)
    starting_state, _ = envs.reset()
    return envs, starting_state

def get_environment_info(envs):



    try:    #For vectorized environments
        state_dim = envs.single_observation_space.shape[0]
        print("See that state_dim, action_dim, n_action and disc/cont is correctly set")
        try:  # discrete
            n_actions = envs.action_space.shape[0]  # env.observation_space.shape
            action_dim = envs.single_action_space.n
            action_type = "discrete"
        except AttributeError:  # continuous
            action_dim = envs.single_action_space.shape[0]
            action_type = "continuous"
            n_actions = envs.action_space.shape[1]
    except:
        action_dim = envs.action_space.shape[0]
        state_dim = envs.observation_space.shape[0]
        action_type = "continuous" #TODO add a way to use simple environment and differentiate between discrete and continous
        n_actions = envs.action_space.shape[0]
    return action_dim, state_dim, n_actions, action_type
