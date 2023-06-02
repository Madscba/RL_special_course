# A class implementing a soft actor critic agent.
import copy
import numpy as np
import torch

from hand_in.agents.base_agent import BaseAgent
from hand_in.models.actor_continuous import ActorNetwork_cont
from hand_in.models.actor_discrete import ActorNetwork_disc
from hand_in.models.SACactor import SACActorNetwork
from hand_in.models.critic_model import CriticNetwork
from hand_in.utils.replay_buffer import ReplayBuffer


# https://www.youtube.com/watch?v=_nFXOZpo50U
# https://www.youtube.com/watch?v=U20F-MvThjM&t=0s
# https://www.youtube.com/watch?v=ioidsRlf79o

# https://arxiv.org/pdf/1812.05905.pdf


class SACAgent_v0(BaseAgent):
    def __init__(self, argparser, action_dim, state_dim, n_actions, action_type, reward_scale: float = 2):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.reward_scale = argparser.args.reward_scale
        self.gamma = argparser.args.gamma
        self.tau = argparser.args.tau
        self.value = CriticNetwork(
            argparser=argparser,
            input_dim=state_dim,
            output_dim=1,
            n_actions=n_actions,
            action_type=action_type,
            name="value"
        )
        self.value_target = CriticNetwork(
            argparser=argparser,
            input_dim=state_dim,
            output_dim=1,
            n_actions=n_actions,
            action_type=action_type,
            name="value_target"
        )
        self.critic_primary = CriticNetwork(
            argparser=argparser,
            input_dim=state_dim + n_actions,
            output_dim=1,
            n_actions=n_actions,
            action_type=action_type,
            name="critic_primary"
        )
        self.critic_secondary = CriticNetwork(
            argparser=argparser,
            input_dim=state_dim + n_actions,
            output_dim=1,
            n_actions=n_actions,
            action_type=action_type,
            name="critic_secondary"
        )
        self.update_value_target(tau=1)

        self.actor_network = SACActorNetwork(
            argparser=argparser, action_dim=action_dim, state_dim=state_dim, name="actor"
        )

        self.replay_buffer = ReplayBuffer(
            capacity=10e5,
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
            used_for_policy_gradient_method=True,
            batch_size=argparser.args.batch_size
        )

    def update_policy(
            self,
            states,
            actions,
            rewards,
            new_states,
            terminated,
            policy_response_dict: dict,
    ):

        if self.replay_buffer.event_idx < self.replay_buffer.batch_size:
            print("nothing happened yet as we have to little exp") if self.replay_buffer.event_idx % 20 == 0 else ""
            return

        event_tuples = self.replay_buffer.get_batch_of_events()
        state, action, reward, state_, terminated = self.event_tuple_to_tensors(event_tuples)
        input_to_critic_networks = torch.cat([state.T, action.T], dim=1)

        self.update_value_network(input_to_critic_networks, state)

        self.update_actor_network(state)

        self.update_critic_networks(input_to_critic_networks, reward, state_, terminated)

        self.update_value_target()

    def update_critic_networks(self, input_to_critic_networks, reward, state_, terminated):
        critic_losses = self.get_critic_losses(input_to_critic_networks, reward, state_, terminated)
        self.critic_primary.optimizer.zero_grad()
        self.critic_secondary.optimizer.zero_grad()
        critic_losses.backward()
        self.critic_primary.optimizer.step()
        self.critic_secondary.optimizer.step()

    def get_critic_losses(self, input_to_critic_networks, reward, state_, terminated):
        reward = reward.view(-1)
        value_ = self.value_target(state_.T).view(-1)
        terminated_idx = np.where(terminated.cpu().numpy() == 1)[0]
        value_[terminated_idx] = 0
        q_hat = self.reward_scale * reward + self.gamma * value_
        q1_old_policy = self.critic_primary.forward(input_to_critic_networks).view(-1)
        q2_old_policy = self.critic_secondary.forward(input_to_critic_networks).view(-1)
        critic_primary_loss = self.critic_primary.criterion(q1_old_policy, q_hat)
        critic_secondary_loss = self.critic_secondary.criterion(q2_old_policy, q_hat)
        critic_losses = critic_primary_loss + critic_secondary_loss
        return critic_losses

    def update_actor_network(self, state):
        actor_loss = self.get_actor_loss(state)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network.optimizer.step()
        self.actor_network.lr_scheduler.step()

    def get_actor_loss(self, state):
        actions, log_probs, _ = self.actor_network.sample_normal(state.T, reparameterize=True)
        log_probs = log_probs.view(-1)
        inp_ = torch.cat([state.T, actions], dim=1)
        q1_new_policy = self.critic_primary.forward(inp_)
        q2_new_policy = self.critic_secondary.forward(inp_)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        return actor_loss

    def update_value_network(self, input_to_critic_networks, state):
        value_loss = self.get_value_loss(input_to_critic_networks, state)
        self.value.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

    def get_value_loss(self, input_to_critic_networks, state):
        value = self.value(state.T).view(-1)
        _, log_probs, _ = self.actor_network.sample_normal(state.T, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_primary.forward(input_to_critic_networks)
        q2_new_policy = self.critic_secondary.forward(input_to_critic_networks)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        value_target = critic_value - log_probs
        value_loss = 0.5 * self.value.criterion(value, value_target)
        return value_loss

    def event_tuple_to_tensors(self, event_tuples):
        states, actions, rewards, new_states, terminated = (
            event_tuples[:self.state_dim, :],  # states
            event_tuples[
            self.state_dim: self.state_dim + self.replay_buffer.n_actions, :
            ],  # actions
            event_tuples[
            self.state_dim
            + self.replay_buffer.n_actions: self.state_dim
                                            + self.replay_buffer.n_actions
                                            + 1,
            :,
            ],  # reward
            event_tuples[
            self.state_dim
            + self.replay_buffer.n_actions
            + 1: 2 * self.state_dim
                 + self.replay_buffer.n_actions
                 + 1,
            :,
            ],  # next state
            event_tuples[
            2 * self.state_dim + self.replay_buffer.n_actions + 1, :
            ],  # terminated
        )
        states, actions, rewards, new_states, terminated = states.to(self.actor_network.device), actions.to(
            self.actor_network.device), rewards.to(self.actor_network.device), new_states.to(
            self.actor_network.device), terminated.to(self.actor_network.device)
        return states, actions, rewards, new_states, terminated

    def follow_policy(self, state, reparameterize=False):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state]).to(self.actor_network.device)

        actions, log_probs, entropy = self.actor_network.sample_normal(state, reparameterize=False)

        policy_response_dict = {
            "log_probs": log_probs,
            "entropy": entropy,
        }
        if actions.shape[0] == self.replay_buffer.batch_size:  # returning batched actions in correct format
            return actions.cpu().detach().numpy(), policy_response_dict
        else:
            return actions.cpu().detach().numpy()[0], policy_response_dict



    def update_value_target(self, tau=1):
        target_value_params = self.value_target.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.value_target.load_state_dict(value_state_dict)

    def uses_replay_buffer(self):
        return True

    def save_models(self):
        print("saving SAC_v0 models:")
        self.critic_primary.save_model_checkpoint()
        self.critic_secondary.save_model_checkpoint()
        self.critic_target_primary.save_model_checkpoint()
        self.critic_target_secondary.save_model_checkpoint()
        self.actor_network.save_model_checkpoint()

    def load_models(self):
        print("loading  SAC_v0 models:")
        self.critic_primary.load_model_checkpoint()
        self.critic_secondary.load_model_checkpoint()
        self.critic_target_primary.load_model_checkpoint()
        self.critic_target_secondary.load_model_checkpoint()
        self.actor_network.load_model_checkpoint()
