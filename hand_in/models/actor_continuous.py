import torch


class ActorNetwork_cont(torch.nn.Module):
    """Actor network (continuous action space)"""

    def __init__(
        self,
        argparser,
        state_dim,
        action_dim,
        # action_upper_bound: float = 1,
        # action_lower_bound: float = -1,
    ):
        super(ActorNetwork_cont, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.n_env
        self.lr = argparser.args.lr
        #self.action_space_range = action_upper_bound - action_lower_bound
        #self.action_space_center = self.action_space_range / 2

        self.continuous = True

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.log_std = torch.nn.Parameter(torch.ones(self.n_envs, self.output_dim))
        self.fc = torch.nn.Linear(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.Linear):
    #         torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self, x):
        mu = self.model(torch.Tensor(x))
        log_sigma_sq = (self.fc(x) + self.log_std)
        log_sigma_sq = torch.clamp(log_sigma_sq, -20, 2)
        sigma_sq = torch.exp(log_sigma_sq)
        dist = torch.distributions.Normal(mu, sigma_sq)
        action = dist.sample()
        entropy = (
            dist.entropy()
        )  # Corresponds to 0.5 * ((log_sigma_sq ** 2 * 2 * pi).log() + 1)  # Entropy of gaussian: https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
        log_probs = dist.log_prob(action)
        return action, log_probs, entropy, {"mu": mu, "sigma_sq": sigma_sq}
