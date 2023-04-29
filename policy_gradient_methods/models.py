import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.autograd import Variable


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
            torch.nn.Linear(hidden_dim, self.output_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        value = self.model(torch.Tensor(x))
        return value


class ActorNetwork(torch.nn.Module):
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
        super(ActorNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = envs.action_space.shape[0]
        self.action_space_range = action_upper_bound - action_lower_bound
        self.action_space_center = self.action_space_range / 2

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.output_dim),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.std = torch.nn.Parameter(torch.ones(self.n_envs,self.output_dim))
    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self, x):
        mu = self.model(torch.Tensor(x))
        sigma_sq = self.std #todo consider adding a more mechanism to ensure std to be positive.
        return mu, sigma_sq

    def predict(self, x):
        """ """
        with torch.no_grad():
            value = self.model_shared(torch.Tensor(x))
            mu = self.model_mu(value)
            sigma_sq = self.model_sigma(value)
            return mu, sigma_sq
