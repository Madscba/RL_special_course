# main.py serves as the entry point for running your RL experiments. It can handle argument parsing, instantiate agents and environments, and orchestrate the training or evaluation process.
from tqdm import tqdm
import numpy as np
import os
import sys
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the repository's root directory to the Python path
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(repo_root)

from hand_in.utils.argparser import Parser
from hand_in.utils.utils import set_seed, evaluate_agent, get_observed_next_state
from hand_in.environment.gym_environment import get_envs
from hand_in.utils.logger import get_logger
from hand_in.utils.utils import get_agent


if __name__ == "__main__":
    p = Parser()
    set_seed(p.args.seed)
    # #
    # for alpha in np.linspace(0.0003,1,3)[::-1]:
    #     for tau in [0.05, 0.005]:
    #         for gamma in [0.99,0.995][::-1]:
    #             for lr in [0.0003, 0.003]:  # lr
    #                 p.args.lr = lr
    #                 p.args.alpha = alpha
    #                 p.args.gamma = gamma
    #                 p.args.tau = tau




                ##


    e, state = get_envs(env_id=p.args.env_name, num_envs=p.args.n_env)
    l = get_logger(p)
    a = get_agent(p, e)

    #################
    # for each episode
    for frame_count in tqdm(range(p.args.n_steps), desc=f"Training"):
        # Follow policy:
        action, policy_response_dict = a.follow_policy(state)
        # take action A and observe rew, new_state
        new_state, reward, terminated, truncated, info = e.step(action)

        observed_next_state = get_observed_next_state(new_state, info)
        if a.uses_replay_buffer():
            a.replay_buffer.save_event(state, action, reward, observed_next_state, terminated, policy_response_dict, info)

        a.update_policy(state, action, reward, observed_next_state, terminated, policy_response_dict)

        state = new_state

        l.log_step(reward)

        if terminated or l.current_episode_frame_counter > 1000:
            # state, _ = e.reset()
            terminated = False
            l.log_episode()
            if l.episode_counter % 200 == 0:
                evaluate_agent(a, p.args.env_name, num_episodes=3)



        # if frame_count % (p.args.n_steps//20) == 0 and (p.args.n_steps/1.25) < frame_count:
        #     evaluate_agent(a, p.args.env_name, num_episodes=1, render=True)

    l.plot_epi_rewards()
    l.plot_step_rewards()
