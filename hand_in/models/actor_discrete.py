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
        #        action_upper_bound: float = 1,
        #        action_lower_bound: float = -1,
    ):
        super(ActorNetwork_disc, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.n_env
        self.lr = argparser.args.lr
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
