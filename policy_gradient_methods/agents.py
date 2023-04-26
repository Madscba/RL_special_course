import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from episode_buffer import EpisodeBuffer
import torch.nn.functional as F
from torch.autograd import Variable
from math import pi


class PolicyGradientAgent:
    def __init__(self, envs, actor_network, parser, critic_network=None):
        self.envs = envs
        self.actor_network = actor_network
        self.parser = parser
        self.n_env = parser.args.n_environments
        self.critic_network = critic_network

    def set_train_mode(self):
        self.actor_network.train()
        if self.critic_network:
            self.critic_network.train()

    def learn_policy(
            self,
            n_frames: int = 2000,
            eval_mode: bool = False,
            learning_algorithm: str = "REINFORCE",
    ):
        self.set_train_mode()
        reward_in_episodes = np.zeros(10000, dtype=float)
        frames_in_episodes = np.zeros(10000, dtype=int)
        episodes = 0
        cur_epi_frame_counter = 0
        cur_epi_reward = 0
        terminated = np.array([False])

        states, _ = self.envs.reset()
        # Algorithmic steps
        # for each episode
        for frame_count in tqdm(
                range(n_frames), desc=f"Frames (training :{not eval_mode}):"
        ):
            if frame_count % 20000 == 0 and frame_count > 0:
                self.save_model(self.actor_network)

            # Generate episode / steps

            episode_history, termination_step, step_info = self.episode_rollout(
                states, learning_algorithm=learning_algorithm
            )
            self.update_model(
                episode_history, termination_step, learning_algo=learning_algorithm, step_info=step_info
            )
            if learning_algorithm == "REINFORCE":
                cur_epi_reward = (
                        np.sum(
                            [
                                episode_history[
                                idx,
                                self.actor_network.input_dim
                                + self.actor_network.output_dim,
                                :term_step,
                                ]
                                .sum()
                                .detach()
                                .numpy()
                                for idx, term_step in enumerate(termination_step)
                            ]
                        )
                        / self.n_env
                )
                cur_epi_frame_counter = episode_history.shape[-1]
                print(f"\n epi: {episodes}, epi_rew: {cur_epi_reward}")
            else:
                pass
                states, actions, rewards, new_states, terminated, log_probs, entropy = step_info
                # new_states, reward, terminated = step_info #todo figure out how to add step number info to limit maximum episode length
                states = new_states
                cur_epi_frame_counter += 1
                cur_epi_reward += rewards.sum()
            if (
                    terminated.any()
                    or cur_epi_frame_counter > 1000
                    or learning_algorithm == "REINFORCE"
            ):
                states, _ = self.envs.reset()
                # terminated = False
                reward_in_episodes[episodes] = cur_epi_reward / self.n_env
                frames_in_episodes[episodes] = cur_epi_frame_counter
                print(
                    f"\n ep: {episodes}, reward: {cur_epi_reward}"
                ) if episodes % 100 == 0 else "",
                cur_epi_frame_counter = 0
                cur_epi_reward = 0
                episodes += 1
                # if episodes > 100 and episodes % 20 == 0 and terminated:
                #     print(f'\nepi: {episodes}, rew: {cur_epi_reward}, frames {cur_epi_frame_counter}')
                #     if np.mean(reward_in_episodes[:episodes][-100:]) > 250:
                #         print("Average goal of 250 has been reached, and training is terminated")
                #         break

        self.visualize_episode_statistics(
            episodes, eval_mode, frames_in_episodes, n_frames, reward_in_episodes
        )

    def episode_rollout(
            self, states, learning_algorithm: str = "REINFORCE", n_steps: int = 5
    ):
        if learning_algorithm == "REINFORCE":  # continue until termination
            states, _ = self.envs.reset()
            episode_termination_step = np.repeat(9999, self.envs.action_space.shape[0])
            episode_terminated = np.repeat(0, self.envs.action_space.shape[0])
            epi_buffer = EpisodeBuffer(self.envs)
            episode_step = 0
            while not episode_terminated.all():
                actions, log_probs, entropy = self.follow_policy(states)
                new_states, rewards, terminated, truncated, info = self.envs.step(
                    actions
                )
                epi_buffer.save_event(
                    states, actions, rewards, new_states, terminated, log_probs, entropy
                )
                states = new_states
                if terminated.any():
                    termination_step_tmp = terminated * np.repeat(
                        episode_step, self.envs.action_space.shape[0]
                    )
                    episode_termination_step[terminated] = np.min(
                        [episode_termination_step, termination_step_tmp], axis=0
                    )[terminated]
                    episode_terminated[terminated] += 1
                episode_step += 1
            episode_hist = epi_buffer.get_episode_hist()
            return episode_hist, episode_termination_step, _
        elif learning_algorithm == "AC":

            actions, log_probs, entropy = self.follow_policy(states)
            new_states, rewards, terminated, truncated, info = self.envs.step(
                actions
            )

            return None, None, (states, actions, rewards, new_states, terminated, log_probs, entropy)

    # def update_model(self,episode_history, termination_step,learning_algo):
    #     with torch.autograd.set_detect_anomaly(True):
    #         if learning_algo == "REINFORCE":
    #             gamma = 0.99
    #             # gammas = torch.Tensor([gamma**t for t in range(np.max(termination_step))])
    #             # G = torch.Tensor(0)
    #             # loss = Variable(torch.Tensor(0), requires_grad=True)
    #             #For each episode rollout
    #             for idx, terminated_at in enumerate(termination_step):
    #                 rel_episode_hist = episode_history [idx,:,:terminated_at]
    #                 rel_rewards = rel_episode_hist[self.model.input_dim+self.model.output_dim,:]
    #                 log_prob_idx = 2*self.model.input_dim+self.model.output_dim+2
    #                 rel_log_probs = rel_episode_hist[log_prob_idx:log_prob_idx+2,:]
    #                 G = torch.zeros(1, 1)
    #                 loss = Variable(torch.Tensor([0]))
    #                 #for each step in an episode
    #                 for t in reversed(range(terminated_at)):
    #                     G = gamma * G + rel_rewards[t]
    #                     loss = loss - (rel_log_probs[:,t] * Variable(G)).sum()
    #                     # G = G + gammas[t] * rel_rewards[t]
    #                     # loss = loss - gammas[t] * (rel_log_probs[:, t] * G).sum()
    #                     # G = torch.dot(gammas[:t+1], rel_rewards[:t+1])
    #                     # loss = gammas[t] * (rel_log_probs[:,t] * G).sum() #- (0.0001 * entropies[i].cuda()).sum()
    #
    #                 # loss = loss / terminated_at
    #                 self.model.optimizer.zero_grad()
    #                 loss.backward()
    #                 #utils.clip_grad_norm(self.model.parameters(), 40)
    #                 self.model.optimizer.step()
    #         # pass        retain_graph=True, accumulate_grad=True

    def update_model(self, episode_history, termination_step, learning_algo, step_info):
        # with torch.autograd.set_detect_anomaly(True):
        if learning_algo == "REINFORCE":
            gamma = 0.99
            for idx, terminated_at in enumerate(termination_step):
                rel_episode_hist = episode_history[idx, :, :terminated_at].clone()
                rel_rewards = rel_episode_hist[
                              self.actor_network.input_dim + self.actor_network.output_dim, :
                              ].clone()
                log_prob_idx = (
                        2 * self.actor_network.input_dim
                        + self.actor_network.output_dim
                        + 2
                )
                rel_log_probs = rel_episode_hist[
                                log_prob_idx: log_prob_idx + 2, :
                                ].clone()
                G = torch.zeros(1, 1, requires_grad=False)
                loss = Variable(
                    torch.Tensor([0]), requires_grad=True
                )  # reset loss to zero
                policy_loss = []
                # print(f"{idx}:, {terminated_at}, {rel_rewards[0]}, {rel_log_probs[0,0]}")
                for t in reversed(range(terminated_at)):
                    G = gamma * G + rel_rewards[t]
                    loss = loss - (rel_log_probs[:, t] * G).sum()
                    # log_prob = rel_log_probs[:, t].clone()  # create a copy of the tensor before applying sum()
                    # loss = loss - (log_prob * G).sum()

                    # policy_loss.append(-rel_log_probs[:, t] * rel_rewards[t])

                # policy_loss = torch.cat(policy_loss).sum()
                # self.model.optimizer.zero_grad()
                # policy_loss.backward()
                # self.model.optimizer.step()
                #
            self.actor_network.optimizer.zero_grad()
            loss.backward()  # set retain_graph to True
            self.actor_network.optimizer.step()
        elif learning_algo == "AC":
            states, actions, rewards, new_states, terminated, log_probs, entropy = step_info
            reward = torch.Tensor(rewards).reshape(-1, 1) + self.parser.args.gamma * self.critic_network(
                torch.Tensor(new_states)) * torch.tensor((1 - terminated.reshape(-1, 1)))
            critic_value_est = self.critic_network(torch.Tensor(states))

            value_loss = self.critic_network.criterion(critic_value_est.clone(), reward.clone())
            action_loss = -(log_probs.squeeze(0) * (reward - critic_value_est).detach()).sum()  # deleted - sign

            if self.parser.args.entropy:
                action_loss += (entropy.squeeze(0)).sum()

            self.critic_network.optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.critic_network.optimizer.step()

            self.actor_network.optimizer.zero_grad()
            action_loss.backward()
            self.actor_network.optimizer.step()

    def pol_REINFORCE(self):
        pass

    def pol_AC(self):
        pass

    def follow_policy(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        mu, sigma_sq = self.actor_network(
            state_tensor
        )
        sigma_sq = F.softplus(sigma_sq)  # cannot have negative std. dev.

        if sigma_sq.isnan().any() or mu.isnan().any():
            a = 2
        dist = torch.distributions.Normal(mu, sigma_sq)
        action = dist.sample()
        entropy = 0.5 * ((
                                     sigma_sq ** 2 * 2 * pi).log() + 1)  # Entropy of gaussian: https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
        if dist.log_prob(action).isnan().any():
            a = 2
        return action.numpy().squeeze(0), dist.log_prob(action), entropy

    def save_model(self, model):
        torch.save(self.actor_network.state_dict(), f"env_lunar.pt")

    def visualize_episode_statistics(
            self,
            episodes,
            eval_mode,
            frames_in_episodes,
            n_frames,
            reward_in_episodes,
            gamma=0.99,
            n_periods=500,
    ):
        periods = np.array_split(frames_in_episodes[:episodes], n_periods)
        average_steps = [np.mean(period) for period in periods]
        plt.plot(list(range(n_periods)), average_steps)
        plt.xlabel("average frames")
        plt.xticks(
            np.linspace(0, n_periods, 10),
            [
                f"{10 * n_frames // n_periods * i}:{10 * n_frames // n_periods * (i + 1)}"
                for i in range(10)
            ],
            rotation=60,
        )
        plt.title(
            f'Average episode length, lr:{self.actor_network.optimizer.defaults["lr"] * 100},n_frames:{n_frames}, gamma: {gamma}'
        )
        plt.subplots_adjust(bottom=0.2)
        plt.show()
        reward_periods = np.array_split(reward_in_episodes[:episodes], n_periods)
        average_rewards = [np.mean(reward_period) for reward_period in reward_periods]
        plt.plot(list(range(n_periods)), average_rewards)
        plt.xlabel("average frames")
        plt.xticks(
            np.linspace(0, n_periods, 10),
            [
                f"{10 * n_frames // n_periods * i}:{10 * n_frames // n_periods * (i + 1)}"
                for i in range(10)
            ],
            rotation=60,
        )
        plt.title(
            f'Average reward pr. episode, , lr:{self.actor_network.optimizer.defaults["lr"] * 100},n_frames:{n_frames}, gamma: {gamma}'
        )
        plt.subplots_adjust(bottom=0.2)
        plt.show()
