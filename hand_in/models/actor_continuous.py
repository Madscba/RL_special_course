import torch
import os
import numpy as np
from torch.distributions.normal import Normal


class ActorNetwork_cont(torch.nn.Module):
    """Actor network (continuous action space)"""

    def __init__(
        self,
        argparser,
        state_dim,
        action_dim,
        name,
        # action_upper_bound: float = 1, #Potentially needed to scale actions if action space is not [-1,1]
    ):
        super(ActorNetwork_cont, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.n_envs = argparser.args.n_env
        self.hidden_dim = argparser.args.hidden_size
        self.lr = argparser.args.lr
        self.checkpoint_dir = os.path.join(os.getcwd(),'results/temporary',name+"_actor_c")
        self.reparam_noise = 1e-6

        #self.action_space_range = action_upper_bound - action_lower_bound
        #self.action_space_center = self.action_space_range / 2

        self.continuous = True

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            #torch.nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.mu_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.sigma_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

        #self.log_std = torch.nn.Parameter(torch.ones(self.n_envs, self.output_dim))
        #self.log_std_layer = torch.nn.Linear(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #    self.optimizer, gamma=0.99
        #)

        # self.apply(self._init_weights)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight.data)
            #torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.model(torch.Tensor(x))

        mu = self.mu_layer(x)
        sigma_sq = self.sigma_layer(x)
        sigma_sq = torch.clamp(sigma_sq, min=self.reparam_noise, max=0.999999)
        # print(mu,sigma_sq)

        dist = Normal(mu, sigma_sq)
        action = dist.sample()
        entropy = dist.entropy()  # Corresponds to 0.5 * ((log_sigma_sq ** 2 * 2 * pi).log() + 1)  # Entropy of gaussian: https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/

        log_probs = dist.log_prob(action)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise) #We don't want value to be 0, so we add a small number (from paper appendix)
        #log_probs = log_probs #we need a single number to match the scalar loss but it will be handled later on

        # mean_value = self.get_mean_parameter_value(self)
        # print("Mean parameter value:", mean_value.item())

        return action, log_probs, entropy, {"mu": mu, "sigma_sq": sigma_sq, "dist": dist}


    def save_model_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


    def get_mean_parameter_value(self, model):
        parameters = model.parameters()
        total_params = 0
        total_value = 0

        for param in parameters:
            total_params += param.numel()
            total_value += torch.sum(param.data)

        mean_value = total_value / total_params
        return mean_value