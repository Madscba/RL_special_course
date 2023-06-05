# A class implementing the Deep Q-Network (DQN) agent.
import copy
import numpy as np
import torch
from torch.autograd import Variable
from hand_in.agents.base_agent import BaseAgent
from hand_in.utils.replay_buffer import ReplayBuffer
from hand_in.models.dqn_model import DQNetwork
from hand_in.environment.gym_environment import get_envs


class DQNAgent(BaseAgent):
    def __init__(
        self, argparser, use_DDQN: bool, state_dim: int, action_dim: int, n_actions: int
    ):
        self.DQN = DQNetwork(
            argparser=argparser, state_dim=state_dim, action_dim=action_dim, name ="DQN"
        )
        if use_DDQN:
            self.target_DQN = DQNetwork(
                argparser=argparser, state_dim=state_dim, action_dim=action_dim, name ="target_DQN"
            )
        else:
            self.target_DQN = 0
        self.parser = argparser
        self.eps = argparser.args.eps
        self.eps_decay = argparser.args.eps_decay
        self.min_eps = argparser.args.min_eps
        self.gamma = argparser.args.gamma
        self.batch_size = argparser.args.batch_size
        self.use_replay = argparser.args.use_replay
        if self.use_replay:
            self.replay_buffer = ReplayBuffer(
                batch_size=self.batch_size,
                state_dim=self.DQN.input_dim,
                n_actions=n_actions,
            )
        self.sample_env, _ = get_envs(
            env_id=argparser.args.env_name, num_envs=1
        )  # Used to sample random actions

    def initialize_policy(self):
        pass

    def update_policy(
        self, state, action, reward, new_state, terminated, empty_dict: dict = {}
    ):
        if not self.use_replay:
            self.update_DQN(state, action, reward, new_state, terminated)
        else:
            self.update_DQN_replay()

    def follow_policy(self, state):
        """eps-greedy policy"""
        if np.random.random() < self.eps:
            action = self.sample_env.action_space.sample()
        else:
            action = np.array([torch.argmax(self.DQN.predict(state)).item()])
        return action, {}

    def save_models(self):
        pass

    def load_models(self):
        pass

    def update_DQN(self, state, action, reward, new_state, terminated):
        q_target = (
            reward
            + (1 - terminated)
            * self.gamma
            * torch.max(self.DQN.predict(new_state)).item()
        )
        q_values_grad = self.DQN(state).squeeze()[action]

        loss = self.DQN.criterion(q_values_grad, torch.Tensor(q_target))
        self.DQN.optimizer.zero_grad()
        loss.backward()
        self.DQN.optimizer.step()
        if self.eps_decay > 0:
            self.eps = max(self.eps - self.eps_decay, self.min_eps)

    def update_DQN_replay(self):
        if self.replay_buffer.event_idx >= self.batch_size * 20:
            state_dim = self.DQN.input_dim

            # Choose which network to update
            if self.target_DQN:
                network = self.target_DQN
            else:
                network = self.DQN

            # Fetch events
            event_tuples = self.replay_buffer.get_batch_of_events()
            states, actions, rewards, new_states, terminated = (
                event_tuples[:state_dim, :],  # states
                event_tuples[
                    state_dim : state_dim + self.replay_buffer.n_actions, :
                ],  # actions
                event_tuples[
                    state_dim
                    + self.replay_buffer.n_actions : state_dim
                    + self.replay_buffer.n_actions
                    + 1,
                    :,
                ],  # reward
                event_tuples[
                    state_dim
                    + self.replay_buffer.n_actions
                    + 1 : 2 * state_dim
                    + self.replay_buffer.n_actions
                    + 1,
                    :,
                ],  # next state
                event_tuples[
                    2 * state_dim + self.replay_buffer.n_actions + 1, :
                ],  # terminated
            )

            # Calculate q_values
            q_values = self.DQN(states.T.unsqueeze(1))
            new_q_values = network.predict(new_states.T.unsqueeze(1))
            target = (
                torch.multiply(
                    torch.max(new_q_values, axis=2).values.squeeze(),
                    (1 - terminated.squeeze()),
                )
                * self.gamma
                + rewards.squeeze()
            )

            loss = self.DQN.criterion(
                q_values.squeeze()[torch.arange(self.batch_size), actions.squeeze().long()],target)
            self.DQN.optimizer.zero_grad()
            loss.backward()
            # for p in self.DQN.parameters():
            #     p.grad.data.clamp_(-1, 1)
            self.DQN.optimizer.step()
            if self.eps_decay > 0:
                self.eps = max(self.eps - self.eps_decay, self.min_eps)

            if (
                self.replay_buffer.event_idx % 1000 == 0
                and self.replay_buffer.event_idx > 0
            ):
                # self.DQN.scheduler.step()
                if self.target_DQN:
                    self.target_DQN = copy.deepcopy(self.DQN)
                    print("copied weights. Loss:", loss)
                    # self.target_DQN.load_state_dict(self.DQN.state_dict())

    def uses_replay_buffer(self):
        return self.use_replay

    def learn_policy(self, n_frames: int = 10000, eval_mode: bool = False):
        reward_in_episodes = np.zeros(10000, dtype=float)
        frames_in_episodes = np.zeros(10000, dtype=int)
        episodes = 0
        cur_epi_frame_counter = 0
        cur_epi_reward = 0

        self.DQN.train()
        if self.target_DQN:
            self.target_DQN.train()

        state, _ = self.env.reset()
        # Algorithmic steps
        # for each episode
        for frame_count in tqdm(
            range(n_frames), desc=f"Frames (training :{not eval_mode}):"
        ):
            if frame_count % 20000 == 0 and frame_count > 0:
                torch.save(self.DQN.state_dict(), f"env_lunar.pt")
                print(
                    f"Saved at frame {frame_count}, episode {episodes}, eps: {self.eps}"
                )
                print(
                    f'eps:{self.eps:.4f}, lr:{self.DQN.optimizer.defaults["lr"]:.2f},latest rew {cur_epi_reward}'
                )
            # for each step in episode
            # choose A(s) from S using policy derived from Q (e.g., eps-greedy)
            ## we will ask NN to evaluate all possible actions (left/right)
            # Epsilon greedy strategy:
            action = self.eps_greedy_policy(state)
            # take action A and observe rew, new_state
            # take action and observe reward and new state.
            new_state, reward, terminated, truncated, info = self.env.step(action)
            # Save history (replay buffer) - not needed for DQN
            self.replay_buffer.save_event(state, action, reward, new_state, terminated)
            # self.update_DQN(state, action, reward, new_state, terminated)
            self.update_DQN_replay()
            state = new_state
            cur_epi_frame_counter += 1
            cur_epi_reward += reward
            if terminated or cur_epi_frame_counter > 1000:
                state, _ = self.env.reset()
                terminated = False
                reward_in_episodes[episodes] = cur_epi_reward
                frames_in_episodes[episodes] = cur_epi_frame_counter
                print(
                    f"\n ep: {episodes}, reward: {cur_epi_reward}"
                ) if episodes % 400 == 0 else "",
                cur_epi_frame_counter = 0
                cur_epi_reward = 0
                episodes += 1
            if episodes > 100 and episodes % 20 == 0 and terminated:
                print(
                    f"\nepi: {episodes}, rew: {cur_epi_reward}, frames {cur_epi_frame_counter}"
                )
                if np.mean(reward_in_episodes[:episodes][-100:]) > 250:
                    print(
                        "Average goal of 250 has been reached, and training is terminated"
                    )
                    break
