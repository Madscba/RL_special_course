import os

import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from gymnasium.wrappers import RecordVideo,RecordEpisodeStatistics,TimeLimit


class ANN_simple(torch.nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim,lr):
        super(ANN_simple,self).__init__()
        self.model = torch.nn.Sequential(
                                        torch.nn.Linear(state_dim,hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, hidden_dim*2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim*2,action_dim),
                                        # torch.nn.Softmax()
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def forward(self,x):
        return self.model(torch.Tensor(x))

class ReplayBuffer():
    def __init__(self,capacity:int=200000, state_dim:int=4,batch_size:int=64):
        self.capacity = capacity
        self.event_idx = 0
        self.event_tuples = np.zeros((2*state_dim+3,capacity))
        # ('state','action','reward',next_state','terminated')
        self.batch_size = batch_size
    def save_event(self,state,action,reward,next_state,terminated):
        self.event_tuples[:,self.event_idx%self.capacity] = np.hstack((state,action,reward,next_state,terminated))
        self.event_idx += 1

    def get_batch_of_events(self):
        if self.event_idx >= self.batch_size:
            sample_idx = np.random.choice(np.min([self.event_idx,self.capacity]), self.batch_size, replace=False)
            return self.event_tuples[:, sample_idx]

class DQN(torch.nn.Module):
    def __init__(self,state_dim,action_dim, lr):
        self.input_dim = state_dim
        self.output_dim = action_dim
        super(DQN, self).__init__()
        self.model = torch.nn.Sequential(
                                        torch.nn.Linear(self.input_dim,256),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(256, 256),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(256,self.output_dim),
                                        # torch.nn.Softmax()
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def forward(self,x):
        return self.model(torch.Tensor(x))


class DQN_agent():
    def __init__(self,env,DQN,eps:float=1,eps_decay:bool=False,min_eps:float=0.05,gamma:float=0.95,batch_size:int=1):
        self.env = env
        self.DQN = DQN
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(batch_size=batch_size, state_dim=self.DQN.input_dim)
    def eps_greedy_policy(self,state):
        q_values = self.DQN.forward(state)
        if np.random.random() < self.eps:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(q_values))
        return action, q_values
    def update_DQN(self):
        n = self.DQN.input_dim
        if self.replay_buffer.event_idx >= self.batch_size:
            event_tuples = self.replay_buffer.get_batch_of_events()
            states,actions,rewards,new_states,terminated = event_tuples[:n,:], event_tuples[n:n+1,:], event_tuples[n+1:n+2,:], event_tuples[n+2:2*n+2,:],event_tuples[2*n+2:,:]

            self.DQN.optimizer.zero_grad()
            q_values = self.DQN.forward(states.T)
            new_q_values = self.DQN.forward(new_states.T)
            target = (torch.multiply(torch.max(new_q_values,axis=1).values,torch.tensor(terminated).squeeze()) * self.gamma + torch.tensor(rewards.squeeze())).float()
            loss = self.DQN.criterion(q_values[torch.arange(self.batch_size),actions].squeeze(), target)
            loss.backward()
            self.DQN.optimizer.step()
            self.eps = np.max([self.eps - 0.00001, self.min_eps])
            # print(self.eps)


    def learn_policy(self,n_episodes:int=100,eval_mode:bool=False):
        steps_in_episodes = np.zeros(n_episodes, dtype=int)
        self.DQN.train()
# Algorithmic steps
#for each episode
        for episode_idx in tqdm(range(n_episodes),desc=f"Episode (training :{not eval_mode}):"):
            if episode_idx%500==0 and episode_idx>0:
                torch.save(self.DQN.state_dict(), f'env_lunar_hidden_hidden_train_w_replay_from_scratch.pt')
                print(f"Saved at episode {episode_idx}")


            # get reset env and get initial state
            new_state, _ = self.env.reset()
            # env._max_episode_steps = 1000
            terminated = False
#for each step in episode
            counter = 0
            while not terminated and counter < 1000:
                state = new_state
#choose A(s) from S using policy derived from Q (e.g., eps-greedy)
## we will ask NN to evaluate all possible actions (left/right)
                #Epsilon greedy strategy:
                action, q_values = self.eps_greedy_policy(state)
#take action A and observe rew, new_state
                #take action and observe reward and new state.
                new_state, reward, terminated, truncated, info = self.env.step(action)

                #Save history (replay buffer) - not needed for DQN
                self.replay_buffer.save_event(state, action, reward, new_state, terminated)

                self.update_DQN()
                counter += 1
                if counter%300 == 0:
                    print(episode_idx, " counter: ",counter)
            steps_in_episodes[episode_idx] = wrapped_env.return_queue[-1]

        n_periods = 100
        periods = np.array_split(steps_in_episodes, n_periods)
        average_steps = [np.mean(period) for period in periods]
        plt.plot(list(range(n_periods)),average_steps)
        plt.xlabel('average steps')
        plt.xticks(np.linspace(0,n_periods,10),[f'{10*n_episodes//n_periods*i}:{10*n_episodes//n_periods*(i+1)}' for i in range(10)], rotation=60)
        plt.title(f'Average episode length, lr:{self.DQN.optimizer.defaults["lr"]},n_episodes:{n_episodes},eps:{eps:.2f},\n eps_decay:{self.eps_decay}, gamma: {gamma} eval_mode: {eval_mode}')
        plt.subplots_adjust(bottom=0.2)
        plt.show()


if __name__  ==  "__main__":
    #cartpole, [n_episodes, n_hidden=10,lr=0.0005,eps=0.3,gamma=0.9]
    # simple anna [6000,n_hidden=20,lr=0.0001,eps=0.85,gamma = 0.9]
    pre_trained_DQN = False
    save_DQN = True
    n_episodes = 3000
    env_type = "lunar"

    if env_type == "cartpole":
        env = gym.make('CartPole-v1')
    else:
        env = gym.make("LunarLander-v2")
    env.reset(seed=41)
    wrapped_env = RecordEpisodeStatistics(env,5000)
    # n_hidden = 20
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    lr = 0.0001
    eps = 1
    gamma = 0.99
    batch_size = 64

    DQN = DQN(n_state,n_action,lr)
    if pre_trained_DQN:
        try:
            DQN.load_state_dict(torch.load(f'env_lunar_hidden_hidden_train_w_replay.pt'))
            print("loaded DQN")
        except:
            print("no pretrained DQN available")
            pass


    # n_episodes = 2000
    q_learning = DQN_agent(wrapped_env,DQN,eps=eps,eps_decay=True,gamma=gamma,batch_size=batch_size)
    start_time = time.time()
    q_learning.learn_policy(n_episodes=n_episodes)
    print(f'Time for {n_episodes}: {time.time()-start_time}')

    if save_DQN:
        torch.save(DQN.state_dict(), f'env_lunar_hidden_hidden_train_w_replay_from_scratch.pt')

    #eval post training
    # n_episodes = 100
    # q_learning = Q_learning(wrapped_env,agent)
    # start_time = time.time()
    # q_learning.learn_policy(n_episodes=n_episodes,eval_mode=True)
    # print(f'Time for {n_episodes}: {time.time()-start_time}')


    if env_type == "cartpole":
        env = gym.make('CartPole-v1', render_mode="rgb_array")
    else:
        env = gym.make("LunarLander-v2", render_mode="rgb_array")

    env = RecordVideo(env,video_folder=os.getcwd(),episode_trigger=lambda x: True)
    n_episodes = 5
    q_learning = DQN_agent(env, DQN,eps=eps,eps_decay=True,gamma=gamma)
    start_time = time.time()
    q_learning.learn_policy(n_episodes=n_episodes, eval_mode=True)
    #
    # if env_type == "cartpole":
    #     env = gym.make('CartPole-v1', render_mode="human")
    # else:
    #     env = gym.make("LunarLander-v2", render_mode="human")
    #
    # n_episodes = 100
    # q_learning = Q_learning(wrapped_env, DQN,eps=eps,eps_decay=True,gamma=gamma)
    # start_time = time.time()
    # q_learning.learn_policy(n_episodes=n_episodes, eval_mode=True)
    # print(f'Time for {n_episodes}: {time.time() - start_time}')
