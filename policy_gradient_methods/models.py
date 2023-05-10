import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.autograd import Variable
import pytest

class CriticNetwork(torch.nn.Module):
    """Critic network"""

    def __init__(self, state_dim, action_dim,envs, hidden_dim: int = 64, lr: float = 0.001):
        super(CriticNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = envs.action_space.shape[0]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss() #MSELoss()

    def forward(self, x):
        value = self.model(torch.Tensor(x))
        return value


class ActorNetwork_cont(torch.nn.Module):
    """Actor network (continuous action space)"""

    def __init__(
        self,
        state_dim,
        action_dim,
        envs,
        hidden_dim: int = 64,
        lr: float = 0.001,
        action_upper_bound: float = 1,
        action_lower_bound: float = -1,
    ):
        super(ActorNetwork_cont, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = envs.action_space.shape[0]
        self.action_space_range = action_upper_bound - action_lower_bound
        self.action_space_center = self.action_space_range / 2
        self.continuous = True

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.output_dim),
        )
        self.std = torch.nn.Parameter(torch.ones(self.n_envs,self.output_dim))
        self.fc = torch.nn.Linear(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self, x):
        mu = self.model(torch.Tensor(x))
        sigma_sq = self.fc(x) + self.std #todo consider adding a more mechanism to ensure std to be positive.

        sigma_sq = torch.clamp(sigma_sq, 0.1, 5)  # cannot have negative std. dev.
        dist = torch.distributions.Normal(mu, sigma_sq)
        action = dist.sample()
        entropy = dist.entropy()  # Corresponds to 0.5 * ((sigma_sq ** 2 * 2 * pi).log() + 1)  # Entropy of gaussian: https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
        log_probs = dist.log_prob(action)
        return action, log_probs,entropy, {'mu': mu, 'sigma_sq': sigma_sq}

    def predict(self, x):
        """ """
        with torch.no_grad():
            mu = self.model(torch.Tensor(x))
            sigma_sq = self.fc(x) + self.std  # todo consider adding a more mechanism to ensure std to be positive.

            sigma_sq = F.softplus(sigma_sq)  # cannot have negative std. dev.
            dist = torch.distributions.Normal(mu, sigma_sq)
            action = dist.sample()
            entropy = dist.entropy()  # Corresponds to 0.5 * ((sigma_sq ** 2 * 2 * pi).log() + 1)  # Entropy of gaussian: https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
            log_probs = dist.log_prob(action)
            return action, log_probs,entropy, {'mu': mu, 'sigma_sq': sigma_sq}


class ActorNetwork_disc(torch.nn.Module):
    """Actor network (discrete action space)"""
    def __init__(
            self,
            state_dim,
            action_dim,
            envs,
            hidden_dim: int = 64,
            lr: float = 0.001,
            action_upper_bound: float = 1,
            action_lower_bound: float = -1,
    ):
        super(ActorNetwork_disc, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = envs.action_space.shape[0]
        self.action_space_range = action_upper_bound - action_lower_bound
        self.action_space_center = self.action_space_range / 2
        self.continuous = False

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.output_dim),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self, x):
        x = self.model(torch.Tensor(x))
        action_probs = F.softmax(x)
        dist = Categorical(action_probs)
        assert abs(float(action_probs.sum()) - round(float(action_probs.sum()),0)) < 0.01
        action = dist.sample()
        entropy = dist.entropy()
        log_probs = dist.log_prob(action)
        return action, log_probs, entropy, {}

    def predict(self, x):
        """ """
        with torch.no_grad():
            x = self.model(torch.Tensor(x))
            action_probs = F.softmax(x)
            dist = Categorical(action_probs)

            action = dist.sample()
            entropy = dist.entropy()
            log_probs = dist.log_prob(action)
            return action, log_probs, entropy, {}
