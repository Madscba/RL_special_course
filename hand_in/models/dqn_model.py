import torch
import os

class DQNetwork(torch.nn.Module):
    def __init__(self, argparser, state_dim, action_dim,name):
        super(DQNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.checkpoint_dir = os.path.join(os.getcwd(),'results/temporary',name+"_DQN")

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.output_dim),
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), argparser.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, x):
        return self.model(torch.Tensor(x))

    def predict(self, state):
        """Compute Q values for all actions using the DQL."""
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def save_model_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
