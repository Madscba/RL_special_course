import torch
import numpy as np

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
        self.log_std_layer = torch.nn.Linear(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #    self.optimizer, gamma=0.99
        #)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.uniform_(module.weight.data)
            #torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        mu = self.model(torch.Tensor(x))
        log_sigma_sq = (self.log_std_layer(x) + self.log_std)
        log_sigma_sq = torch.clamp(log_sigma_sq, -20, 2)
        sigma_sq = torch.exp(log_sigma_sq)
        dist = torch.distributions.Normal(mu, sigma_sq)
        action = dist.sample()
        entropy = (
            dist.entropy()
        )  # Corresponds to 0.5 * ((log_sigma_sq ** 2 * 2 * pi).log() + 1)  # Entropy of gaussian: https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/

        value = torch.clamp(action, -0.999999, 0.999999)
        pre_tanh_value = torch.log(1 + value) / 2 - torch.log(1 - value) / 2

        log_probs = dist.log_prob(pre_tanh_value)

        correction_log_prob = - 2. * (
                torch.tensor(0.69314718,dtype=torch.float32) #torch.from_numpy(np.log[2.])
                - pre_tanh_value
                - torch.nn.functional.softplus(-2. * pre_tanh_value)
        )

        return action, log_probs + correction_log_prob, entropy, {"mu": mu, "sigma_sq": sigma_sq}
