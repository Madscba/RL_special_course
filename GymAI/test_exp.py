import requests
import numpy as np
import math
from scipy.stats import expon,binom,poisson,norm

#Double q-learning experiment
# Test whether we should follow policy of Q1 and then do update based on Q2 or the other way around.

class QLearning_off_policy():
    def __init__(self,eps:float = 0.1, alpha:float=0.2):
        self.alpha = alpha
        self.eps = eps


class DoubleQLearning():
    def __init__(self, eps: float = 0.1):
        self.Q_table = [0,0]
        self.eps = eps

class DoubleQLearning_greedy_wrt_itself(DoubleQLearning):
    pass


def exp_double_Qlearning(episodes: int = 10,mu: float = -0.1, std: float = 1,alpha: float = 0.1,eps: float = 0.1) -> None:
    r_A, r_B = 0,  np.random.normal(loc=mu, scale=std, size=(1, episodes))




if __name__ == "__main__":
    # double_q_learning()
