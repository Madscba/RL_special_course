# A class implementing a soft actor critic agent.
import copy
import numpy as np
import torch

from hand_in.agents.base_agent import BaseAgent
from hand_in.models.actor_continuous import ActorNetwork_cont
from hand_in.models.actor_discrete import ActorNetwork_disc
from hand_in.models.critic_model import CriticNetwork
from hand_in.utils.replay_buffer import ReplayBuffer

# https://www.youtube.com/watch?v=_nFXOZpo50U
# https://www.youtube.com/watch?v=U20F-MvThjM&t=0s
# https://www.youtube.com/watch?v=ioidsRlf79o

# https://arxiv.org/pdf/1812.05905.pdf


class SACAgent(BaseAgent):
    def __init__(self, argparser, action_dim, state_dim, n_actions, action_type,reward_scale:float=2):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.reward_scale = argparser.args.reward_scale
        self.gamma = argparser.args.gamma
        self.tau = argparser.args.tau
        self.log_alpha = torch.tensor(np.log(torch.tensor(argparser.args.alpha,dtype=torch.float32)),dtype=torch.float32, requires_grad=True) #We introduce and learn log alpha to ensure positivity and exponentiate it whenever needed
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.parser.args.lr)
        self.target_entropy = torch.tensor(-action_dim,dtype=torch.float32, requires_grad=True)
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
        self.critic_target_primary = CriticNetwork(
            argparser=argparser,
            input_dim=state_dim + n_actions,
            output_dim=1,
            n_actions=n_actions,
            action_type=action_type,
            name="critic_t_1"
        )
        self.critic_target_secondary = CriticNetwork(
            argparser=argparser,
            input_dim=state_dim + n_actions,
            output_dim=1,
            n_actions=n_actions,
            action_type=action_type,
            name="critic_t_2"
        )
        self.initialize_targets()

        self.actor_network = self.init_actor_network(
            argparser, action_dim, state_dim, n_actions, action_type
        )
        self.replay_buffer = ReplayBuffer(
            capacity=10e5,
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
            used_for_policy_gradient_method=True,
            batch_size=argparser.args.batch_size
        )

    def initialize_targets(self):
        self.critic_target_primary = self.initialize_target(self.critic_primary)
        self.critic_target_secondary = self.initialize_target(self.critic_secondary)
    def initialize_target(self, network_to_be_copied):
        target_network = copy.deepcopy(network_to_be_copied)
        for p in target_network.parameters():
            p.requires_grad = False
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
        else:
            if self.replay_buffer.event_idx % (self.replay_buffer.batch_size * 10) == 0:
                print(
                    f" mu: {policy_response_dict['mu']}, std: {policy_response_dict['sigma_sq']}, log_probs: {policy_response_dict['sigma_sq']}, entropy: {policy_response_dict['entropy']}") #alpha: {self.log_alpha.exp()},

        event_tuples = self.replay_buffer.get_batch_of_events()
        actions, new_states, rewards, states, terminated = self.event_tuple_to_tensors(event_tuples)

        # losses = self.compute_losses(states, actions, rewards, new_states, terminated)

        #### Update Q function networks #####
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
        critic_value_prim = self.critic_primary(state_action_tensor.clone().detach())
        critic_value_sec = self.critic_secondary(state_action_tensor.clone().detach())

        value_loss_prim = 0.5 * self.critic_primary.criterion(
            critic_value_prim.squeeze(2), critic_target
        )
        value_loss_sec = 0.5 * self.critic_secondary.criterion(
            critic_value_sec.squeeze(2), critic_target
        )

        self.critic_primary.optimizer.zero_grad()
        self.critic_secondary.optimizer.zero_grad()

        value_losses = value_loss_prim + value_loss_sec
        value_losses.backward()
        # value_loss_prim.backward()
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

        # action_loss = losses['action_loss']
        self.actor_network.optimizer.zero_grad()
        action_loss.backward()
        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [p for g in self.actor_network.optimizer.param_groups for p in g["params"]],
        #         self.parser.args.grad_clipping,)
        self.actor_network.optimizer.step()

        # for params in self.critic_primary.parameters():
        #     params.requires_grad = True
        # for params in self.critic_secondary.parameters():
        #     params.requires_grad = True




        # Update alpha:
        # alpha_loss = losses["alpha_loss"]
        # print("alpha is not being updated")
        # self.alpha_optim.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optim.step()
        # self.alpha = self.log_alpha.exp()


        #### Update Q function target networks with exponentially moving average:
        for param, target_param in zip(self.critic_primary.parameters(), self.critic_target_primary.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic_secondary.parameters(),
            self.critic_target_secondary.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

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
        return actions, new_states, rewards, states, terminated

    def follow_policy(self, state, reparameterize=False):
        if not isinstance(state,torch.Tensor):
            state = torch.from_numpy(state).float().to(self.actor_network.device)

        action, log_probs, entropy, info_dict = self.actor_network(state)
        dist = info_dict['dist']

        if reparameterize:
            actions = dist.rsample() #gradients can be used if rsample method is used.
        else:
            actions = dist.sample()

        action = torch.tanh(actions)
        log_probs = dist.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.actor_network.reparam_noise)
        log_probs = log_probs #.sum(1, keepdim=True)


        if self.continuous:
            #print("potentially worth logging mu and std")
            mu = info_dict["mu"]
            sigma_sq = info_dict["sigma_sq"]
            policy_response_dict = {
                "mu": mu,
                "sigma_sq": sigma_sq,
                "log_probs": log_probs,
                "entropy": entropy,
            }
        else:
            policy_response_dict = {"log_probs": log_probs, "entropy": entropy}
        return action.cpu().detach().numpy(), policy_response_dict

    def compute_losses(self,states, actions, rewards, new_states, terminated):
        # Q-functions loss
        actions_next, policy_response_dict_next = self.follow_policy(new_states.T.unsqueeze(1),reparameterize=False)
        log_probs_next = policy_response_dict_next["log_probs"]
        new_state_action_tensor = torch.hstack(
            (new_states.T, torch.Tensor(actions_next).reshape(self.replay_buffer.batch_size, -1).to(self.actor_network.device))).unsqueeze(1)
        critic_value_target_min = torch.min(
            self.critic_target_secondary(new_state_action_tensor),
            self.critic_target_secondary(new_state_action_tensor),
        )
        reward = torch.Tensor(rewards).reshape(-1, 1)

        critic_target = self.reward_scale*reward + self.gamma * torch.tensor(1 - terminated.reshape(-1, 1)) * (
                    critic_value_target_min.squeeze(1) - (self.log_alpha.exp() * log_probs_next).mean(axis=2))
        # add .mean(axis=2) (log_alpha.exp * log_probs)?

        state_action_tensor = torch.Tensor(torch.vstack((states, actions))).T.unsqueeze(1)
        critic_value_prim = self.critic_primary(state_action_tensor.clone().detach())
        critic_value_sec = self.critic_secondary(state_action_tensor.clone().detach())

        value_loss_prim = 0.5 * self.critic_primary.criterion(
            critic_value_prim.squeeze(2), critic_target
        )
        value_loss_sec = 0.5 * self.critic_secondary.criterion(
            critic_value_sec.squeeze(2), critic_target
        )

        #alpha and policy loss
        new_obs_actions, policy_response_dict = self.follow_policy(states.T.unsqueeze(1), reparameterize=True)
        log_probs = policy_response_dict['log_probs'].unsqueeze(-1)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()

        state_new_obs_action_tensor = torch.hstack((states.T.clone(), torch.Tensor(new_obs_actions).reshape(self.replay_buffer.batch_size,-1).to(self.actor_network.device))).unsqueeze(1)
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

    def init_actor_network(
        self, argparser, action_dim, state_dim, n_actions, action_type
    ):
        if self.continuous:
            return ActorNetwork_cont(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim, name="actor"
            )
        else:
            return ActorNetwork_disc(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim
            )
