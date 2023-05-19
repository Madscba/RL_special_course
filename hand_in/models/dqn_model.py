import torch
class DQNetwork(torch.nn.Module):
    def __init__(self,argparser,state_dim,action_dim):
        super(DQNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.output_dim),
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), argparser.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        return self.model(torch.Tensor(x))

    def predict(self, state):
        """Compute Q values for all actions using the DQL."""
        with torch.no_grad():
            return self.model(torch.Tensor(state))
