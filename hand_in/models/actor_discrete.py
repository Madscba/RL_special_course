import os
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNetwork_disc(torch.nn.Module):
    """Actor network (discrete action space)"""

    def __init__(
        self,
        argparser,
        state_dim,
        action_dim,
        name,
    ):
        super(ActorNetwork_disc, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.hidden_size
        self.checkpoint_file = os.path.join(
            os.getcwd(), "results/temporary", name + "_actor_d"
        )
        self.lr = argparser.args.lr
        self.continuous = False

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_dim, self.hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        #     torch.nn.ReLU(),
        #     # torch.nn.Linear(self.hidden_dim, self.output_dim),
        # )
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        self.action_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.value_layer = torch.nn.Linear(self.hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"
        self.to(self.device)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # x = self.model(torch.Tensor(x))
        action_values = self.action_layer(x)
        state_values = self.value_layer(x)
        return (action_values, state_values)

    def get_action_and_log_prob(self, state):
        action_values, state_values = self.forward(state)
        action_probs = F.softmax(action_values, dim=2)
        dist = Categorical(action_probs)
        action = dist.sample()
        entropy = dist.entropy()
        log_probs = dist.log_prob(action)
        return action, log_probs, entropy, {}

    def save_model_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file + ".pt")

    def load_model_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file + ".pt"))
