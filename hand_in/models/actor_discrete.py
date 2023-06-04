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
        #        action_upper_bound: float = 1,
        #        action_lower_bound: float = -1,
    ):
        super(ActorNetwork_disc, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.hidden_size
        self.checkpoint_file = os.path.join(os.getcwd(),'results/temporary',name+"_actor_d")
        self.lr = argparser.args.lr
        self.reparam_noise = 1e-6
        # self.action_space_range = action_upper_bound - action_lower_bound
        # self.action_space_center = self.action_space_range / 2
        self.continuous = False

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.to(self.device)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self, x):
        x = self.model(torch.Tensor(x))
        action_probs = F.softmax(x)
        dist = Categorical(action_probs)
        assert (
            abs(float(action_probs.sum()) - round(float(action_probs.sum()), 0)) < 0.01
        )
        action = dist.sample()
        entropy = dist.entropy()
        log_probs = dist.log_prob(action)
        return action, log_probs, entropy, {}

    def save_model_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
