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
            argparser=argparser, state_dim=state_dim, action_dim=action_dim, name="DQN"
        )
        self.use_DDQN = use_DDQN
        if self.use_DDQN:
            self.target_DQN = DQNetwork(
                argparser=argparser,
                state_dim=state_dim,
                action_dim=action_dim,
                name="target_DQN",
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
                q_values.squeeze()[
                    torch.arange(self.batch_size), actions.squeeze().long()
                ],
                target,
            )
            self.DQN.optimizer.zero_grad()
            loss.backward()

            self.DQN.optimizer.step()
            if self.eps_decay > 0:
                self.eps = max(self.eps - self.eps_decay, self.min_eps)

            if (
                self.replay_buffer.event_idx % 1000 == 0
                and self.replay_buffer.event_idx > 0
            ):
                if self.target_DQN:
                    self.target_DQN = copy.deepcopy(self.DQN)
                    print("copied weights. Loss:", loss)

    def uses_replay_buffer(self):
        return self.use_replay

    def save_models(self):
        print("saving DQN/DDQN models:")
        if self.use_DDQN:
            self.target_DQN.save_model_checkpoint()
        self.DQN.save_model_checkpoint()

    def load_models(self):
        print("loading DQN/DDQN models:")
        if self.use_DDQN:
            self.target_DQN.load_model_checkpoint()
        self.DQN.load_model_checkpoint()
