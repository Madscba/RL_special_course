import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import ExponentialLR


class SACActorNetwork(torch.nn.Module):
    def __init__(self, argparser, state_dim, action_dim, name):
        super(SACActorNetwork, self).__init__()
        self.input_dim = state_dim
        self.hidden_dim = argparser.args.hidden_size
        self.output_dim = action_dim
        self.name = name
        self.lr = argparser.args.lr
        self.reparam_noise = 1e-6
        self.checkpoint_file = os.path.join(
            os.getcwd(), "results/temporary", name + "_SACActor"
        )

        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mu = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.sigma = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)

        if reparameterize:
            actions = dist.rsample()
        else:
            actions = dist.sample()

        action = torch.tanh(actions).to(self.device)
        log_probs = dist.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        entropy = dist.entropy()

        return action, log_probs, entropy

    def save_model_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file + ".pt")

    def load_model_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file + ".pt"))
