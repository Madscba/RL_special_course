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


class SACAgent_v1(BaseAgent):
    def __init__(self, argparser, action_dim, state_dim, n_actions, action_type, reward_scale: float = 2):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.reward_scale = argparser.args.reward_scale
        self.gamma = argparser.args.gamma
        self.tau = argparser.args.tau
        self.log_alpha = torch.tensor(np.log(torch.tensor(argparser.args.alpha, dtype=torch.float32)),
                                      dtype=torch.float32,
                                      requires_grad=True)  # We introduce and learn log alpha to ensure positivity and exponentiate it whenever needed
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.parser.args.lr)
        self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32, requires_grad=True)
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
            name="critic_1"
        )
        self.critic_secondary = CriticNetwork(
            argparser=argparser,
            input_dim=state_dim + n_actions,
            output_dim=1,
            n_actions=n_actions,
            action_type=action_type,
            name="critic_2"
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

    def initialize_target(self, network_to_be_copied):
        target_network = copy.deepcopy(network_to_be_copied)
        # for p in target_network.parameters():
        #     p.requires_grad = False
        return target_network

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

        if True:
            self.learn()
            return
        event_tuples = self.replay_buffer.get_batch_of_events()
        state, action, reward, new_state, terminated = self.event_tuple_to_tensors(event_tuples)

        ## Update value networks
        state_value = self.value(state.T).view(-1)
        next_state_value = self.value_target(new_state.T).view(-1)
        terminated_idx = np.where(terminated.cpu().numpy() == 1)[0]
        next_state_value[terminated_idx] = 0

        actions, policy_response_dict = self.follow_policy(state.T, reparameterize=False)

        log_probs = policy_response_dict['log_probs'].sum(1)
        new_state_action_tensor = torch.hstack(
            (state.T, torch.Tensor(actions).reshape(self.replay_buffer.batch_size, -1).to(
                self.actor_network.device)))
        critic_value_min = torch.min(
            self.critic_primary(new_state_action_tensor),
            self.critic_secondary(new_state_action_tensor),
        ).view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value_min - log_probs * self.log_alpha.exp()
        value_loss = 0.5 * self.value.criterion(state_value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        #### Update Q function networks #####
        reward = reward.view(-1)
        q_target = self.reward_scale * reward + self.gamma * next_state_value

        # actions_next, policy_response_dict_next = self.follow_policy(new_states.T.unsqueeze(1), reparameterize=False)
        # log_probs_next = policy_response_dict_next['log_probs'].sum(2,keepdim=True).squeeze(1)
        state_action_tensor = torch.hstack(
            (state.T, torch.Tensor(action).reshape(self.replay_buffer.batch_size, -1).to(
                self.actor_network.device)))
        critic_value_prim = self.critic_primary(state_action_tensor).view(-1)
        critic_value_sec = self.critic_secondary(state_action_tensor).view(-1)

        critic_loss_prim = 0.5 * self.critic_primary.criterion(
            critic_value_prim, q_target
        )
        critic_loss_sec = 0.5 * self.critic_secondary.criterion(
            critic_value_sec, q_target
        )

        # self.critic_primary.optimizer.zero_grad()
        # self.critic_secondary.optimizer.zero_grad()

        critic_losses = critic_loss_prim + critic_loss_sec
        critic_losses.backward()
        # value_loss_prim.backward(retain_graph=True)
        # value_loss_sec.backward()

        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [
        #             p
        #             for g in self.critic_primary.optimizer.param_groups
        #             for p in g["params"]
        #         ],
        #         self.parser.args.grad_clipping,
        #     )
        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [
        #             p
        #             for g in self.critic_secondary.optimizer.param_groups
        #             for p in g["params"]
        #         ],
        #         self.parser.args.grad_clipping,
        #     )

        self.critic_primary.optimizer.step()
        self.critic_secondary.optimizer.step()

        #### Update actor network
        # for params in self.critic_primary.parameters():
        #     params.requires_grad = False
        # for params in self.critic_secondary.parameters():
        #     params.requires_grad = False
        actions, policy_response_dict_ = self.follow_policy(state.T, reparameterize=True)
        log_probs = policy_response_dict_['log_probs'].sum(1)

        state_action_tensor = torch.hstack(
            (state.T, torch.Tensor(actions).reshape(self.replay_buffer.batch_size, -1).to(
                self.actor_network.device)))
        critic_value_min = torch.min(
            self.critic_primary(state_action_tensor),
            self.critic_secondary(state_action_tensor)
        ).view(-1)
        # action_loss = (self.log_alpha.exp() * log_probs - critic_value_min).mean(0)

        # action_loss = losses['action_loss']
        action_loss = (self.log_alpha.exp() * log_probs - critic_value_min).mean(0)
        self.actor_network.optimizer.zero_grad()
        action_loss.backward(retain_graph=True)
        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [p for g in self.actor_network.optimizer.param_groups for p in g["params"]],
        #         self.parser.args.grad_clipping,)
        self.actor_network.optimizer.step()

        # for params in self.critic_primary.parameters():
        #     params.requires_grad = True
        # for params in self.critic_secondary.parameters():
        #     params.requires_grad = True

        self.update_value_target(tau=self.tau)
        # print(f'val: {value_loss},c1: {critic_loss_prim}, c2 {critic_loss_sec}, po: {action_loss}' )
        # Update alpha:
        # alpha_loss = losses["alpha_loss"]
        # print("alpha is not being updated")
        # new_obs_actions, policy_response_dict = self.follow_policy(states.T.unsqueeze(1), reparameterize=True)
        # log_probs = policy_response_dict['log_probs'].sum(2,keepdim=True).squeeze(1)
        # alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean(0)
        #
        # self.alpha_optim.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optim.step()
        # self.alpha = self.log_alpha.exp()

        #### Update Q function target networks with exponentially moving average:

        # for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
        #     target_param.data.copy_(
        #         self.tau * param.data + (1 - self.tau) * target_param.data
        #     )

        # for param, target_param in zip(
        #     self.critic_secondary.parameters(),
        #     self.critic_target_secondary.parameters(),
        # ):
        #     target_param.data.copy_(
        #         self.tau * param.data + (1 - self.tau) * target_param.data
        #     )

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

    def learn(self):
        event_tuples = self.replay_buffer.get_batch_of_events()
        state, action, reward, state_, terminated = self.event_tuple_to_tensors(event_tuples)


        reward = reward.view(-1)
        value = self.value(state.T).view(-1)
        value_ = self.value_target(state_.T).view(-1)
        terminated_idx = np.where(terminated.cpu().numpy() == 1)[0]
        value_[terminated_idx] = 0

        actions, log_probs, _ = self.actor_network.sample_normal(state.T, reparameterize=False)
        log_probs = log_probs.view(-1)
        inp = torch.cat([state.T, action.T], dim=1)
        q1_new_policy = self.critic_primary.forward(inp)
        q2_new_policy = self.critic_secondary.forward(inp)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * self.value.criterion(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs, _ = self.actor_network.sample_normal(state.T, reparameterize=True)
        log_probs = log_probs.view(-1)
        inp_ = torch.cat([state.T, actions], dim=1)
        q1_new_policy = self.critic_primary.forward(inp_)
        q2_new_policy = self.critic_secondary.forward(inp_)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network.optimizer.step()

        self.critic_primary.optimizer.zero_grad()
        self.critic_secondary.optimizer.zero_grad()
        q_hat = self.reward_scale * reward + self.gamma * value_
        q1_old_policy = self.critic_primary.forward(inp).view(-1)
        q2_old_policy = self.critic_secondary.forward(inp).view(-1)
        critic_1_loss = 0.5 * self.critic_primary.criterion(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * self.critic_secondary.criterion(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_primary.optimizer.step()
        self.critic_secondary.optimizer.step()

        self.update_value_target()

    def compute_losses(self, states, actions, rewards, new_states, terminated):
        # Q-functions loss
        actions_next, policy_response_dict_next = self.follow_policy(new_states.T.unsqueeze(1), reparameterize=False)
        log_probs_next = policy_response_dict_next["log_probs"]
        new_state_action_tensor = torch.hstack(
            (new_states.T, torch.Tensor(actions_next).reshape(self.replay_buffer.batch_size, -1).to(
                self.actor_network.device))).unsqueeze(1)
        critic_value_target_min = torch.min(
            self.critic_target_secondary(new_state_action_tensor),
            self.critic_target_secondary(new_state_action_tensor),
        )
        reward = torch.Tensor(rewards).reshape(-1, 1)

        critic_target = self.reward_scale * reward + self.gamma * torch.tensor(1 - terminated.reshape(-1, 1)) * (
                critic_value_target_min.squeeze(1) - (self.log_alpha.exp() * log_probs_next).mean(axis=2))
        # add .mean(axis=2) (log_alpha.exp * log_probs)?

        state_action_tensor = torch.Tensor(torch.vstack((states, actions))).T.unsqueeze(1)
        critic_value_prim = self.critic_primary(state_action_tensor)
        critic_value_sec = self.critic_secondary(state_action_tensor)

        value_loss_prim = 0.5 * self.critic_primary.criterion(
            critic_value_prim.squeeze(2), critic_target
        )
        value_loss_sec = 0.5 * self.critic_secondary.criterion(
            critic_value_sec.squeeze(2), critic_target
        )

        # alpha and policy loss
        new_obs_actions, policy_response_dict = self.follow_policy(states.T.unsqueeze(1), reparameterize=True)
        log_probs = policy_response_dict['log_probs'].unsqueeze(-1)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()

        state_new_obs_action_tensor = torch.hstack((states.T.clone(),
                                                    torch.Tensor(new_obs_actions).reshape(self.replay_buffer.batch_size,
                                                                                          -1).to(
                                                        self.actor_network.device))).unsqueeze(1)
        critic_value_min = torch.min(
            self.critic_primary(state_new_obs_action_tensor),
            self.critic_secondary(state_new_obs_action_tensor),
        )
        action_loss = (self.log_alpha.exp() * log_probs - critic_value_min.clone()).mean()
        # action_loss = (log_probs - critic_value_min).mean()
        # print("consider whether  alpha should be multiplied on or not")

        losses = {"alpha_loss": alpha_loss,
                  "action_loss": action_loss,
                  "value_loss_prim": value_loss_prim,
                  "value_loss_sec": value_loss_sec
                  }

        return losses

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
        print("saving all 5 models:")
        self.critic_primary.save_model_checkpoint()
        self.critic_secondary.save_model_checkpoint()
        self.critic_target_primary.save_model_checkpoint()
        self.critic_target_secondary.save_model_checkpoint()
        self.actor_network.save_model_checkpoint()

    def load_models(self):
        print("loading all 5 models:")
        self.critic_primary.load_model_checkpoint()
        self.critic_secondary.load_model_checkpoint()
        self.critic_target_primary.load_model_checkpoint()
        self.critic_target_secondary.load_model_checkpoint()
        self.actor_network.load_model_checkpoint()
