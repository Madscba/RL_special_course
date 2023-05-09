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

        #For debugging purposes
        self.episodes = 0
        self.frames = 0
        self.pred_mu = np.zeros((parser.args.n_episodes,2), dtype=float)
        self.pred_sigma_sq = np.zeros((parser.args.n_episodes,2), dtype=float)
        self.value_loss = np.zeros(parser.args.n_episodes, dtype=float)
        self.action_loss = np.zeros(parser.args.n_episodes, dtype=float)
        self.entropy = np.zeros((parser.args.n_episodes,2), dtype=float)
        self.log_prob = np.zeros((parser.args.n_episodes,2), dtype=float)
        self.continuous = self.actor_network.continuous
        if self.continuous:
            self.action_dim = envs.single_action_space.shape[0]
        else:
            self.action_dim = 1 #Only pick a single action

    def set_train_mode(self):
        self.actor_network.train()
        if self.critic_network:
            self.critic_network.train()
    def set_eval_mode(self):
        self.actor_network.eval()
        if self.critic_network:
            self.critic_network.eval()

    def learn_policy(
            self,
            n_frames: int = 2000,
            eval_mode: bool = False,
            learning_algorithm: str = "REINFORCE",
    ):
        self.set_train_mode()
        reward_in_episodes = np.zeros(10000, dtype=float)
        frames_in_episodes = np.zeros(10000, dtype=int)
        self.episodes = 0
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
                                + self.action_dim,
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
                print(f"\n epi: {self.episodes}, epi_rew: {cur_epi_reward}")
            else:
                self.frames = frame_count
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
                reward_in_episodes[self.episodes] = cur_epi_reward / self.n_env
                frames_in_episodes[self.episodes] = cur_epi_frame_counter
                print(f"\n ep: {self.episodes}, reward: {cur_epi_reward}, epi length: {cur_epi_frame_counter}# actions: {actions[0]}"
                ) if self.episodes % 1 == 0 and learning_algorithm !="REINFORCE" else "",
                cur_epi_frame_counter = 0
                cur_epi_reward = 0
                self.episodes += 1

        print(f'visualize {self.parser.args.visualize}')
        if self.parser.args.visualize:
            self.visualize_episode_statistics(
                self.episodes, eval_mode, frames_in_episodes, n_frames, reward_in_episodes, self.parser
            )
        if self.parser.args.visualize:
            print("Plotting debugging graphs")
            plt.plot(np.arange(self.frames), self.pred_mu[:self.frames, 0],label="mu_0")
            plt.plot(np.arange(self.frames), self.pred_mu[:self.frames, 1],label="mu_1")
            plt.title("mu")
            plt.legend()
            plt.show()

            plt.plot(np.arange(self.frames), self.entropy[:self.frames, 1],label="entropy_0")
            plt.plot(np.arange(self.frames), self.entropy[:self.frames, 0],label="entropy_1")
            plt.legend()
            plt.title("Entropies")
            plt.show()

            plt.plot(np.arange(self.frames), self.pred_sigma_sq[:self.frames, 1],label="sigma_sq_0")
            plt.plot(np.arange(self.frames), self.pred_sigma_sq[:self.frames, 0],label="sigma_sq_1")
            plt.title("sigma_sq")
            plt.legend()
            plt.show()

            plt.plot(np.arange(self.frames), self.value_loss[:self.frames],label="value_loss")
            plt.plot(np.arange(self.frames), self.action_loss[:self.frames],label="action_loss")
            plt.title("Losses during frames")
            plt.legend()
            plt.show()
    def episode_rollout(
            self, states, learning_algorithm: str = "REINFORCE", n_steps: int = 5
    ):
        if learning_algorithm == "REINFORCE":  # continue until termination
            states, _ = self.envs.reset()
            episode_termination_step = np.repeat(1002, self.envs.action_space.shape[0])
            episode_terminated = np.repeat(0, self.envs.action_space.shape[0])
            epi_buffer = EpisodeBuffer(envs = self.envs,action_dim = self.action_dim)
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

    def update_model(self, episode_history, termination_step, learning_algo, step_info):
        # with torch.autograd.set_detect_anomaly(True):
        if learning_algo == "REINFORCE":
            for idx, terminated_at in enumerate(termination_step):
                rel_episode_hist = episode_history[idx, :, :terminated_at].clone()
                rel_rewards = rel_episode_hist[
                              self.actor_network.input_dim + self.action_dim, :
                              ].clone()
                log_prob_idx = (
                        2 * self.actor_network.input_dim
                        + self.action_dim
                        + 2) #('state','action','reward',next_state','terminated', log_probs, entropy)


                rel_log_probs = rel_episode_hist[
                                log_prob_idx: log_prob_idx + 2, :
                                ].clone()
                G = torch.zeros(1, 1, requires_grad=False)
                loss = Variable(
                    torch.Tensor([0]), requires_grad=True
                )  # reset loss to zero
                for t in reversed(range(terminated_at)):
                    G = self.parser.args.gamma * G + rel_rewards[t]
                    loss = loss - (rel_log_probs[:, t] * G).sum()

                #
            self.actor_network.optimizer.zero_grad()
            loss.backward()  # set retain_graph to True
            if self.parser.args.grad_clipping:
                torch.nn.utils.clip_grad_norm_([p for g in self.actor_network.optimizer.param_groups for p in g["params"]], 1)
            self.actor_network.optimizer.step()
        elif learning_algo == "AC":
            states, actions, rewards, new_states, terminated, log_probs, entropy = step_info
            reward = torch.Tensor(rewards).reshape(-1, 1) + self.parser.args.gamma * self.critic_network(
                torch.Tensor(new_states)) * torch.tensor((1 - terminated.reshape(-1, 1)))
            critic_value_est = self.critic_network(torch.Tensor(states))

            value_loss = self.critic_network.criterion(critic_value_est.clone(), reward.clone())
            action_loss = -(log_probs.squeeze(0) * (reward - critic_value_est).detach()).sum()  # - appears to work decently


            if self.parser.args.entropy:
                action_loss += 0.5*(entropy.squeeze(0)).sum() #+ appears to work decently or at least better
            #For debugging purposes
            self.value_loss[self.frames] = value_loss.clone().detach().numpy()
            self.action_loss[self.frames] = action_loss.clone().detach().numpy()

            if np.random.uniform() < 0.001:
                print(f"\t val_loss: {round(self.value_loss[self.frames],2)}, action_loss{round(self.action_loss[self.frames],2)}, rew: {np.round(reward.detach().numpy(),1)}, critic_est {np.round(critic_value_est.detach().numpy(),1)}" )

            self.critic_network.optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            if self.parser.args.grad_clipping:
                torch.nn.utils.clip_grad_norm_([p for g in self.critic_network.optimizer.param_groups for p in g["params"]], 1)
            self.critic_network.optimizer.step()

            self.actor_network.optimizer.zero_grad()
            action_loss.backward()
            if self.parser.args.grad_clipping:
                torch.nn.utils.clip_grad_norm_([p for g in self.critic_network.optimizer.param_groups for p in g["params"]], 1)
            self.actor_network.optimizer.step()

    def pol_REINFORCE(self):
        pass

    def pol_AC(self):
        pass

    def follow_policy(self, state):
        state_tensor = torch.from_numpy(state).float()
        action, log_probs, entropy, info_dict= self.actor_network(state_tensor)

        if self.continuous:
            mu = info_dict['mu']
            sigma_sq = info_dict['sigma_sq']
            # For debugging purposes
            self.pred_mu[self.frames] = mu.clone().detach().numpy()
            self.pred_sigma_sq[self.frames] = sigma_sq.clone().detach().numpy()

        #self.log_prob[self.frames] = log_probs.clone().detach().numpy()
        #self.entropy[self.frames] = entropy.clone().detach().numpy()

        return action.numpy(), log_probs, entropy

    def save_model(self, model):
        torch.save(self.actor_network.state_dict(), f"env_lunar.pt")

    def evaluate_policy(
            self,
            n_frames: int = 2000,
            learning_algorithm: str = "REINFORCE",
    ):
        self.set_eval_mode()
        self.learn_policy(n_frames= n_frames,eval_mode=True, learning_algorithm = learning_algorithm)


    def visualize_episode_statistics(
            self,
            episodes,
            eval_mode,
            frames_in_episodes,
            n_frames,
            reward_in_episodes,
            parser,
    ):
        episodes_to_plot = np.min((100,episodes))
        periods = np.array_split(frames_in_episodes[:episodes], episodes_to_plot)
        average_steps = [np.mean(period) for period in periods]
        plt.plot(list(range(episodes_to_plot)), average_steps)
        plt.xlabel("Episodes")
        plt.xticks(
            np.linspace(0, episodes_to_plot, 10),
            [
                f"{episodes_to_plot/10 * i}:{ episodes_to_plot/10 * (i + 1)}"
                for i in range(10)
            ],
            rotation=60,
        )
        plt.title(
            f'Average episode length, lr:{self.actor_network.optimizer.defaults["lr"]},epi:{episodes}, algo: {self.parser.args.learning_algorithm}, \n ent: {self.parser.args.entropy}, grad_cl: {self.parser.args.entropy}, n_env: {self.parser.args.n_environments}'
        )
        plt.subplots_adjust(bottom=0.2)
        plt.show()

        reward_periods = np.array_split(reward_in_episodes[:episodes], episodes_to_plot)
        average_rewards = [np.mean(reward_period) for reward_period in reward_periods]
        plt.plot(list(range(episodes_to_plot)), average_rewards)
        plt.xlabel("average frames")
        plt.xticks(
            np.linspace(0, episodes_to_plot, 10),
            [
                f"{int(episodes_to_plot/10 * i)}:{ int(episodes_to_plot/10 * (i + 1))}"
                for i in range(10)
            ],
            rotation=60,
        )
        plt.title(
            f'Average reward pr. episode, lr:{self.actor_network.optimizer.defaults["lr"] },epi:{episodes}, algo: {self.parser.args.learning_algorithm}, \n ent: {self.parser.args.entropy}, grad_cl: {self.parser.args.entropy}, n_env: {self.parser.args.n_environments}'
        )
        plt.subplots_adjust(bottom=0.2)
        plt.show()
