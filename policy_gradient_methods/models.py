import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim,envs,  hiden_dim: int = 64, lr: float = 0.001, action_upper_bound: float = 1,
                 action_lower_bound: float = -1):
        super(PolicyNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = envs.action_space.shape[0]
        self.action_space_range = action_upper_bound - action_lower_bound
        self.action_space_center = self.action_space_range / 2
        self.model_shared = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hiden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiden_dim, hiden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiden_dim, self.output_dim),
        )
        self.model_mu = torch.nn.Sequential(
            torch.nn.Linear(hiden_dim, self.output_dim),
        )
        self.model_sigma = torch.nn.Sequential(
            torch.nn.Linear(hiden_dim, self.output_dim),
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        value = self.model_shared(torch.Tensor(x))
        #value = (value * 2) - 1
        return F.softmax(value,dim=1)

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            value =  self.model(torch.Tensor(state))
            value = 1 / (1+torch.exp(value))
            #value = torch.sigmoid(value)
            value = value*self.action_space_range-self.action_space_center

            return value
