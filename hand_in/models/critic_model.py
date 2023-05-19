import torch
class CriticNetwork(torch.nn.Module):
    """Critic network"""

    def __init__(
        self, argparser,state_dim,action_dim,n_actions, action_type):
        super(CriticNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.n_env
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=argparser.args.lr, weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss()  # MSELoss()

    def forward(self, x):
        value = self.model(torch.Tensor(x))
        return value