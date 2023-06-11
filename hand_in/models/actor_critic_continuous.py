import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal


class ActorNetwork_cont(torch.nn.Module):
    """Actor network (continuous action space)"""

    def __init__(
        self,
        argparser,
        state_dim,
        action_dim,
        name,
    ):
        super(ActorNetwork_cont, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.hidden_size
        self.lr = argparser.args.lr
        self.checkpoint_file = os.path.join(
            os.getcwd(), "results/temporary", name + "_actor_c"
        )
        self.reparam_noise = 1e-6
        self.continuous = True
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mu = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.log_sigma = torch.nn.parameter.Parameter(torch.Tensor([0.5]))
        self.value = torch.nn.Linear(self.hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        state_value = self.value(prob)
        return (mu, sigma), state_value

    def get_action_and_log_prob(self, state):
        (mu, sigma), state_value = self.forward(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        entropy = dist.entropy()
        log_probs = dist.log_prob(action)
        return torch.tanh(action), log_probs, entropy, {}

    def save_model_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file + ".pt")

    def load_model_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file + ".pt"))
