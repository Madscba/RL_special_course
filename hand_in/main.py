#main.py serves as the entry point for running your RL experiments. It can handle argument parsing, instantiate agents and environments, and orchestrate the training or evaluation process.
from tqdm import tqdm
import numpy as np
from hand_in.utils.argparser import Parser
from hand_in.utils.utils import set_seed
from hand_in.environment.gym_environment import get_envs
from hand_in.utils.logger import get_logger
from hand_in.utils.utils import get_agent

if __name__ == "__main__":
    p = Parser()
    set_seed(p.args.seed)
    e, state = get_envs(env_id=p.args.env_name, num_envs=p.args.n_env)
    print(e)
    l = get_logger(p)
    a = get_agent(p,e)

    #################
    # for each episode
    for frame_count in tqdm(range(p.args.n_steps), desc=f"Training"):
        # Follow policy:
        action, policy_response_dict = a.follow_policy(state)
        # take action A and observe rew, new_state
        new_state, reward, terminated, truncated, info = e.step(action)
        # Save history (replay buffer) - not needed for DQN
        if a.uses_replay_buffer():
            a.replay_buffer.save_event(state, action, reward, new_state, terminated, policy_response_dict)

        a.update_policy(state, action, reward, new_state, terminated,policy_response_dict)
        state = new_state

        l.log_step(reward)

        if terminated or l.current_episode_frame_counter > 1000:
            state, _ = e.reset()
            terminated = False
            l.log_episode()

    l.plot_epi_rewards()
    l.plot_step_rewards()
