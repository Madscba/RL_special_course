# A class implementing a policy gradient-based actor critic agent.
import numpy as np
import torch
from torch.autograd import Variable

from hand_in.agents.base_agent import BaseAgent
from hand_in.models.actor_continuous import ActorNetwork_cont
from hand_in.models.actor_discrete import ActorNetwork_disc

from hand_in.models.critic_model import CriticNetwork
from hand_in.utils.replay_buffer import ReplayBuffer


class ACAgent(BaseAgent):
    def __init__(self, argparser, action_dim, state_dim, n_actions, action_type):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.actor_network = self.init_actor_network(
            argparser, action_dim, state_dim, n_actions, action_type
        )
        self.critic_network = CriticNetwork(
            argparser, state_dim, action_dim, n_actions, action_type, name="critic"
        )
        if self.parser.args.use_replay:
            self.replay_buffer = ReplayBuffer(
                capacity=1005,
                state_dim=state_dim,
                action_dim=action_dim,
                n_actions=n_actions,
                used_for_policy_gradient_method=True,
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
        # (states,actions,rewards,new_states,terminated,log_probs,entropy,) = step_info
        log_probs, entropy = (
            policy_response_dict["log_probs"],
            policy_response_dict["entropy"],
        )
        # dist = policy_response_dict['dist']
        # entropy = policy_response_dict["entropy"]
        # log_probs = dist.log_prob(torch.Tensor(actions))

        # Calculate reward and critic estimate

        critic_value = self.critic_network.forward(torch.Tensor(states))
        critic_value = critic_value.view(-1)
        critic_value_ = self.critic_network.forward(torch.Tensor(new_states))
        critic_value_ = critic_value_.view(-1)
        terminated_idx = np.where(terminated == 1)[0]
        critic_value_[terminated_idx] = 0

        reward = torch.Tensor(rewards).view(-1) + self.parser.args.gamma * critic_value_
        td_err = (reward - critic_value)

        # Calculate losses for actor and critic
        value_loss = td_err**2
        # Update networks
        # losses = value_loss + action_loss
        # losses.backward()
        self.critic_network.optimizer.zero_grad()
        value_loss.backward()
        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [p for g in self.critic_network.optimizer.param_groups for p in g["params"]],
        #         self.parser.args.grad_clipping,
        #     )
        self.critic_network.optimizer.step()

        if self.continuous:
            action_loss = -(log_probs.sum(2).view(-1) * td_err.clone().detach())
        else:
            action_loss = -(log_probs.view(-1) * td_err.clone().detach())
        # add entropy
        if self.parser.args.entropy:
            action_loss += (0.5 * (entropy.squeeze(0)).sum())
        self.actor_network.optimizer.zero_grad()
        action_loss.backward()
        if self.parser.args.grad_clipping:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.actor_network.optimizer.param_groups for p in g["params"]],
                self.parser.args.grad_clipping,
            )
        self.actor_network.optimizer.step()

    def follow_policy(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state]).to(self.actor_network.device)


        action, log_probs, entropy, info_dict = self.actor_network.get_action_and_log_prob(state)
        if self.continuous:
            policy_response_dict = {
                "log_probs": log_probs,
                "entropy": entropy,
            }
        else:
            policy_response_dict = {"log_probs": log_probs, "entropy": entropy}
        return action.numpy()[0], policy_response_dict

    def init_actor_network(
        self, argparser, action_dim, state_dim, n_actions, action_type
    ):
        if self.continuous:
            return ActorNetwork_cont(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim, name="actor"
            )
        else:
            return ActorNetwork_disc(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim, name="actor"
            )

    def save_models(self):
        print("saving actor critic models:")
        self.actor_network.save_model_checkpoint()

    def load_models(self):
        print("loading actor critic models:")
        self.actor_network.load_model_checkpoint()


    def uses_replay_buffer(self):
        """Replay buffer appears to hurt the stability of learning and is hence turned off"""
        return False
