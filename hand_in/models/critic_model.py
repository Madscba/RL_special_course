import torch
class CriticNetwork(torch.nn.Module):
    """Critic network"""

    def __init__(
        self, state_dim, action_dim, envs, hidden_dim: int = 64, lr: float = 0.001
    ):
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss()  # MSELoss()

    def forward(self, x):
        value = self.model(torch.Tensor(x))
        return value