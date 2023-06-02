import os
import torch


class CriticNetwork(torch.nn.Module):
    """Critic network"""

    def __init__(self, argparser, input_dim, output_dim, n_actions, action_type, name):
        super(CriticNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.hidden_size
        self.checkpoint_file = os.path.join(os.getcwd(),'results/temporary',name+"_critic")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=argparser.args.lr)
        self.criterion = torch.nn.MSELoss()  # MSELoss()

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, x):
        value = self.model(torch.Tensor(x))
        return value



    def save_model_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))