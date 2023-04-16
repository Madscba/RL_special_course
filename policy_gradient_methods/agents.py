import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from episode_buffer import EpisodeBuffer


class PolicyGradientAgent():
    def __init__(self, envs, model):
        self.envs = envs
        self.model = model

    def learn_policy(self, n_frames: int = 10000, eval_mode: bool = False,learning_algorithm:str="REINFORCE"):
        reward_in_episodes = np.zeros(10000, dtype=float)
        frames_in_episodes = np.zeros(10000, dtype=int)
        episodes = 0
        cur_epi_frame_counter = 0
        cur_epi_reward = 0

        self.model.train()

        states, _ = self.envs.reset()
        # Algorithmic steps
        # for each episode
        for frame_count in tqdm(range(n_frames), desc=f"Frames (training :{not eval_mode}):"):
            if frame_count % 20000 == 0 and frame_count > 0:
                self.save_model(self.model)

            # Generate episode / steps

            episode_history, termination_step = self.episode_rollout(states,learning_algo=learning_algorithm)
            self.update_model(episode_history, termination_step,learning_algo=learning_algorithm)
            states = new_states
            cur_epi_frame_counter += 1
            cur_epi_reward += reward
            if terminated or cur_epi_frame_counter > 1000:
                states, _ = self.envs.reset()
                terminated = False
                reward_in_episodes[episodes] = cur_epi_reward
                frames_in_episodes[episodes] = cur_epi_frame_counter
                print(f'\n ep: {episodes}, reward: {cur_epi_reward}') if episodes % 400 == 0 else "",
                cur_epi_frame_counter = 0
                cur_epi_reward = 0
                episodes += 1
            if episodes > 100 and episodes % 20 == 0 and terminated:
                print(f'\nepi: {episodes}, rew: {cur_epi_reward}, frames {cur_epi_frame_counter}')
                if np.mean(reward_in_episodes[:episodes][-100:]) > 250:
                    print("Average goal of 250 has been reached, and training is terminated")
                    break

        self.visualize_episode_statistics(episodes, eval_mode, frames_in_episodes, n_frames, reward_in_episodes)

    def episode_rollout(self, states, learning_algorithm: str = "REINFORCE", n_steps: int = 5):
        if learning_algorithm == "REINFORCE":  # continue until termination
            states, _ = self.envs.reset()
            episode_termination_step = np.repeat(9999,self.envs.action_space.shape[0])
            episode_terminated = np.repeat(0,self.envs.action_space.shape[0])
            epi_buffer = EpisodeBuffer(self.envs)
            episode_step = 0
            while not episode_terminated.all():
                actions = self.follow_policy(states)
                new_states, rewards, terminated, truncated, info = self.envs.step(actions)
                epi_buffer.save_event(states, actions, rewards, new_states, terminated)
                states = new_states
                if terminated.any():
                    termination_step_tmp = terminated * np.repeat(episode_step,self.envs.action_space.shape[0])
                    episode_termination_step[terminated] = np.min([episode_termination_step,termination_step_tmp],axis=0)[terminated]
                    episode_terminated[terminated] += 1
                episode_step +=1
            episode_hist = epi_buffer.get_episode_hist()
            return episode_hist, episode_termination_step

    def update_model(self,episode_history, termination_step,learning_algo):
        if learning_algo == "REINFORCE":
            gamma = 0.99
            for idx, terminated_at in enumerate(termination_step):
                rel_episode_hist = episode_history [idx,:,:terminated_at]
                rel_rewards = rel_episode_hist[self.model.state_dim+self.model.action_dim,:]
                for t in range(terminated_at):
                    rew = [gamma**(k-t-1)*rel_rewards[k] for k in range(t,terminated_at)]


        pass

    def pol_REINFORCE(self):
        pass

    def pol_AC(self):
        pass

    def follow_policy(self, state):
        action = self.model.predict(state) #torch.argmax(self.model.predict(state)).item()  # had int cast instead of item()
        return np.array(action)

    def save_model(self, model):
        torch.save(self.model.state_dict(), f'env_lunar.pt')

    def visualize_episode_statistics(self, episodes, eval_mode, frames_in_episodes, n_frames, reward_in_episodes,
                                     n_periods=500):

        periods = np.array_split(frames_in_episodes[:episodes], n_periods)
        average_steps = [np.mean(period) for period in periods]
        plt.plot(list(range(n_periods)), average_steps)
        plt.xlabel('average frames')
        plt.xticks(np.linspace(0, n_periods, 10),
                   [f'{10 * n_frames // n_periods * i}:{10 * n_frames // n_periods * (i + 1)}' for i in range(10)],
                   rotation=60)
        plt.title(
            f'Average episode length, lr:{self.model.optimizer.defaults["lr"] * 100},n_frames:{n_frames},eps:{eps:.2f},\n eps_decay:{self.eps_decay}, gamma: {gamma} eval_mode: {eval_mode}')
        plt.subplots_adjust(bottom=0.2)
        plt.show()
        reward_periods = np.array_split(reward_in_episodes[:episodes], n_periods)
        average_rewards = [np.mean(reward_period) for reward_period in reward_periods]
        plt.plot(list(range(n_periods)), average_rewards)
        plt.xlabel('average frames')
        plt.xticks(np.linspace(0, n_periods, 10),
                   [f'{10 * n_frames // n_periods * i}:{10 * n_frames // n_periods * (i + 1)}' for i in range(10)],
                   rotation=60)
        plt.title(
            f'Average reward pr. episode, lr:{self.model.optimizer.defaults["lr"]},n_frames:{n_frames},eps:{eps:.2f},\n eps_decay:{self.eps_decay:.2f}, gamma: {gamma} eval_mode: {eval_mode}')
        plt.subplots_adjust(bottom=0.2)
        plt.show()
