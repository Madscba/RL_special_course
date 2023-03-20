import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

class ANN_simple(torch.nn.Module):
    #ToDo: Implement initialization of weights.

    #ToDo: Implement dropout: torch.nn.Dropout(p=0.3)
    #ToDo: Implement batch norm: torch.nnBatchNorm1d(4)
    #ToDo: Look into deep residual net architecture
    #ToDo: Investigate CNN
    def __init__(self,state_dim,action_dim,hidden_dim,lr):
        super(ANN_simple,self).__init__()
        self.model = torch.nn.Sequential(
                                        torch.nn.Linear(state_dim,hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, hidden_dim*2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim*2,action_dim),
                                        torch.nn.Softmax())
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def forward(self,x):
        return self.model(torch.Tensor(x))

# class CNN_simple(torch.nn.Module):
#
#     def __init__(self):
#         super(CNN_simple, self).__init__()
#         # 1 input image channel (black & white), 6 output channels, 5x5 square convolution         # kernel
#         self.conv1 = torch.nn.Conv2d(1, 6, 5)
#         self.conv2 = torch.nn.Conv2d(6, 16, 3)
#         # an affine operation: y = Wx + b
#         self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features



class Policy():
    def __init__(self):
        pass


class Q_learning():
    def __init__(self,env,model):
        self.env = env
        self.model = model


    def forward(self,x):
        return self.model.forward(x)

    def learn_policy(self,n_episodes:int=100,eps:float=0.95):
        #load optimizer, load criterion
        self.model.train()
# Algorithmic steps
#for each episode
        self.env.reset(seed=42)
        steps_in_episodes = np.zeros((n_episodes),dtype=int)
        for episode_idx in tqdm(range(n_episodes),desc="Episode:"):
            # get reset env and get initial state
            new_state, _ = self.env.reset()
            terminated = False
#for each step in episode
            counter = 0
            while not terminated:
                state = new_state
                #Save history (replay buffer) - not needed for DQN?
#choose A(s) from S using policy derived from Q (e.g., eps-greedy)
## we will ask NN to evaluate all possible actions (left/right)
                #Epsilon greedy strategy:
                q_values = self.model.forward(state)
                if np.random.random() < eps:
                    #take random action
                    action = env.action_space.sample()
                else:
                    #use NN to get values of each state
                    action = int(torch.argmax(q_values)) #.item()
#take action A and observe rew, new_state
                #take action and observe reward and new state.
                new_state, reward, terminated, truncated, info = self.env.step(action)

                if terminated:
                    #Update using the
                    loss = self.model.criterion(q_values[action],torch.as_tensor(reward))
                    #TODO: update when terminated
                    self.model.optimizer.zero_grad()
                    loss.backward()
                    self.model.optimizer.step()
                    # print(f'steps: {counter}')
                    steps_in_episodes[episode_idx] = counter
                else:
                    new_q_values = self.model.forward(new_state)

                    target = reward + torch.max(new_q_values)
                    loss = self.model.criterion(q_values[action],target)
                    self.model.optimizer.zero_grad()
                    loss.backward()
                    self.model.optimizer.step()
                    #calculate loss
                    #backpropagate
                counter+=1
        plt.plot(list(range(n_episodes)),steps_in_episodes)
        plt.show()

if __name__  ==  "__main__":
    RENDER = False
    n_episodes = 1500

    if RENDER:
        env = gym.make('CartPole-v1',render_mode="human")
    else:
        env = gym.make('CartPole-v1')
    n_hidden = 32
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    lr = 0.95
    #ToDo add learning rate, optimizer and more.
    agent = ANN_simple(n_state,n_action,n_hidden,lr)

    q_learning = Q_learning(env,agent)
    start_time = time.time()
    q_learning.learn_policy(n_episodes=n_episodes)
    print(f'Time for {n_episodes}: {time.time()-start_time}')
