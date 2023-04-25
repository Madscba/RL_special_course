import gymnasium as gym
import numpy as np

from models import ActorNetwork
from agents import PolicyGradientAgent


def get_envs(env_id: str = "LunarLander-v2",
             num_envs: int = 10,
             asynchronous: bool = False,
             continuous: bool = True,
             gravity: float = -10.0,
             enable_wind: bool = False,
             wind_power: float = 15.0,
             turbulence_power: float = 1.5
             ):
    envs = gym.vector.make(
        id=env_id,
        num_envs=num_envs,
        asynchronous=asynchronous,
        continuous=continuous,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power
    )
    return envs


if __name__ == "__main__":
    # Set hyperparams
    num_envs = 20
    # Init environments
    envs = get_envs(num_envs=num_envs)
    _ = envs.reset(seed=42)
    #
    n_state = envs.single_observation_space.shape[0]
    n_action = envs.single_action_space.shape[0]

    model = ActorNetwork(n_state, n_action, envs=envs)
    agent = PolicyGradientAgent(envs=envs, model=model)

    agent.learn_policy(n_frames=200)

    envs.close()
