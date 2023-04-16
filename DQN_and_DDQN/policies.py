import os

import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import time, copy
from torch.autograd import Variable


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
    def __init__(self,capacity:int=100000, state_dim:int=4,batch_size:int=64):
        self.capacity = capacity
        self.event_idx = 0
        self.event_tuples = torch.zeros((2*state_dim+3,capacity))
        # ('state','action','reward',next_state','terminated')
        self.batch_size = batch_size
    def save_event(self,state,action,reward,next_state,terminated):
        self.event_tuples[:,self.event_idx%self.capacity] = torch.from_numpy(np.hstack((state,action,reward,next_state,terminated)))
        self.event_idx += 1
        if self.event_idx % 100000 ==0:
            print("starting new buffer")

    def get_batch_of_events(self):
        if self.event_idx >= self.batch_size:
            sample_idx = np.random.choice(np.min([self.event_idx,self.capacity]), self.batch_size, replace=False)
            return self.event_tuples[:, sample_idx]

class DQNetwork(torch.nn.Module):
    def __init__(self,state_dim,action_dim, lr):
        super(DQNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.model = torch.nn.Sequential(
                                        # torch.nn.Dropout(0.2),
                                        torch.nn.Linear(self.input_dim,64),
                                        torch.nn.ReLU(),
                                        # torch.nn.Dropout(0.1),
                                        torch.nn.Linear(64, 64),
                                        torch.nn.ReLU(),
                                        # torch.nn.Dropout(0.1),
                                        # torch.nn.Linear(32, 32),
                                        # torch.nn.ReLU(),
                                        torch.nn.Linear(64,self.output_dim),
                                        # torch.nn.Softmax()
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self,x):
        return self.model(torch.Tensor(x))
    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

class DQN_agent():
    def __init__(self,env,DQN,target_DQN:bool=False,eps:float=1,eps_decay:float=0,min_eps:float=0.05,gamma:float=0.95,batch_size:int=1):
        self.env = env
        self.DQN = DQN
        self.target_DQN = target_DQN
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(batch_size=batch_size, state_dim=self.DQN.input_dim)
    def eps_greedy_policy(self,state):
        if np.random.random() < self.eps:
            action = env.action_space.sample()
        else:
            action = torch.argmax(self.DQN.predict(state)).item() #had int cast instead of item()
        return action


    def update_DQN(self, state, action, reward, new_state, terminated):

        q_values = self.DQN.predict(state)
        if not terminated:
            q_values_next = self.DQN.predict(new_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
        else:
            q_values[action] = reward

        q_values_grad = self.DQN(state)
        loss = self.DQN.criterion(q_values_grad, Variable(torch.Tensor(q_values)))
        self.DQN.optimizer.zero_grad()
        loss.backward()
        self.DQN.optimizer.step()
        if self.eps_decay > 0:
            self.eps = max(self.eps - self.eps_decay, self.min_eps)

    def update_DQN_replay(self):
        if self.replay_buffer.event_idx >= self.batch_size*20:
            n = self.DQN.input_dim

            #Choose which network to update
            if self.target_DQN:
                network = self.target_DQN
            else:
                network = self.DQN

            #Fetch events
            event_tuples = self.replay_buffer.get_batch_of_events()
            states,actions,rewards,new_states,terminated = event_tuples[:n,:], event_tuples[n:n+1,:], event_tuples[n+1:n+2,:], event_tuples[n+2:2*n+2,:],event_tuples[2*n+2:,:]

            #Calculate q_values
            q_values = self.DQN(states.T.unsqueeze(1))
            new_q_values = network.predict(new_states.T.unsqueeze(1))
            target = torch.multiply(torch.max(new_q_values,axis=2).values.squeeze(),(1-terminated.squeeze())) * self.gamma + rewards.squeeze()


            loss = self.DQN.criterion(q_values.squeeze()[torch.arange(self.batch_size),actions.squeeze().long()], target)
            self.DQN.optimizer.zero_grad()
            loss.backward()
            # for p in self.DQN.parameters():
            #     p.grad.data.clamp_(-1, 1)
            self.DQN.optimizer.step()
            if self.eps_decay > 0:
                self.eps = max(self.eps - self.eps_decay, self.min_eps)

            if self.replay_buffer.event_idx % 1000 == 0 and self.replay_buffer.event_idx>0:
                # self.DQN.scheduler.step()
                if self.target_DQN:
                    self.target_DQN = copy.deepcopy(self.DQN)
                    print("copied weights. Loss:",loss)
                    # self.target_DQN.load_state_dict(self.DQN.state_dict())



    def learn_policy(self,n_frames:int=10000,eval_mode:bool=False):
        reward_in_episodes = np.zeros(10000, dtype=float)
        frames_in_episodes = np.zeros(10000,dtype=int)
        episodes = 0
        cur_epi_frame_counter = 0
        cur_epi_reward = 0

        self.DQN.train()
        if self.target_DQN:
            self.target_DQN.train()

        state, _ = self.env.reset()
# Algorithmic steps
#for each episode
        for frame_count in tqdm(range(n_frames),desc=f"Frames (training :{not eval_mode}):"):
            if frame_count%20000==0 and frame_count>0:
                torch.save(self.DQN.state_dict(), f'env_lunar.pt')
                print(f"Saved at frame {frame_count}, episode {episodes}, eps: {self.eps}")
                print(f'eps:{self.eps:.4f}, lr:{self.DQN.optimizer.defaults["lr"]:.2f},latest rew {cur_epi_reward}')
#for each step in episode
#choose A(s) from S using policy derived from Q (e.g., eps-greedy)
## we will ask NN to evaluate all possible actions (left/right)
            #Epsilon greedy strategy:
            action = self.eps_greedy_policy(state)
#take action A and observe rew, new_state
            #take action and observe reward and new state.
            new_state, reward, terminated, truncated, info = self.env.step(action)
            #Save history (replay buffer) - not needed for DQN
            self.replay_buffer.save_event(state, action, reward, new_state, terminated)
            # self.update_DQN(state, action, reward, new_state, terminated)
            self.update_DQN_replay()
            state = new_state
            cur_epi_frame_counter += 1
            cur_epi_reward += reward
            if terminated or cur_epi_frame_counter > 1000:
                state, _ = self.env.reset()
                terminated = False
                reward_in_episodes[episodes] = cur_epi_reward
                frames_in_episodes[episodes] = cur_epi_frame_counter
                print(f'\n ep: {episodes}, reward: {cur_epi_reward}') if episodes%400==0 else "",
                cur_epi_frame_counter = 0
                cur_epi_reward = 0
                episodes += 1
            if episodes > 100 and episodes % 20 == 0 and terminated:
                print(f'\nepi: {episodes}, rew: {cur_epi_reward}, frames {cur_epi_frame_counter}')
                if np.mean(reward_in_episodes[:episodes][-100:]) > 250:
                    print("Average goal of 250 has been reached, and training is terminated")
                    break

        n_periods = 500
        periods = np.array_split(frames_in_episodes[:episodes], n_periods)
        average_steps = [np.mean(period) for period in periods]
        plt.plot(list(range(n_periods)),average_steps)
        plt.xlabel('average frames')
        plt.xticks(np.linspace(0,n_periods,10),[f'{10*n_frames//n_periods*i}:{10*n_frames//n_periods*(i+1)}' for i in range(10)], rotation=60)
        plt.title(f'Average episode length, lr:{self.DQN.optimizer.defaults["lr"]*100},n_frames:{n_frames},eps:{eps:.2f},\n eps_decay:{self.eps_decay}, gamma: {gamma} eval_mode: {eval_mode}')
        plt.subplots_adjust(bottom=0.2)
        plt.show()

        reward_periods = np.array_split(reward_in_episodes[:episodes], n_periods)
        average_rewards = [np.mean(reward_period) for reward_period in reward_periods]
        plt.plot(list(range(n_periods)),average_rewards)
        plt.xlabel('average frames')
        plt.xticks(np.linspace(0,n_periods,10),[f'{10*n_frames//n_periods*i}:{10*n_frames//n_periods*(i+1)}' for i in range(10)], rotation=60)
        plt.title(f'Average reward pr. episode, lr:{self.DQN.optimizer.defaults["lr"]},n_frames:{n_frames},eps:{eps:.2f},\n eps_decay:{self.eps_decay:.2f}, gamma: {gamma} eval_mode: {eval_mode}')
        plt.subplots_adjust(bottom=0.2)
        plt.show()


if __name__  ==  "__main__":
    #cartpole, [n_episodes, n_hidden=10,lr=0.0005,eps=0.3,gamma=0.9]
    # simple anna [6000,n_hidden=20,lr=0.0001,eps=0.85,gamma = 0.9]
    pre_trained_DQN = False
    save_DQN = True
    DDQN = True
    n_frames = 50000
    env_type = "lunar"

    if env_type == "cartpole":
        env = gym.make('CartPole-v1', render_mode="rgb_array")
    else:
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env.reset(seed=41)
    # wrapped_env = RecordEpisodeStatistics(env,5000)
    wrapped_env = RecordVideo(env,video_folder=os.getcwd(),episode_trigger=lambda x: x%50==0 and x > 500,name_prefix="train_videos_cart")
    # n_hidden = 20
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    lr = 0.0001 #0.001 #0.0001
    eps = 0.6   #0.3 #1
    eps_decay = 0.6/n_frames #0.99995
    gamma = 0.99 #0.9
    batch_size = 64 #20

    DQN = DQNetwork(n_state,n_action,lr)
    if DDQN:
        target_DQN = DQNetwork(n_state, n_action, lr)
    else:
        target_DQN = False

    if pre_trained_DQN:
        try:
            if DDQN:
                if env_type == "cartpole":
                    target_DQN.load_state_dict(torch.load(f'env_lunar.pt'))
                else:
                    target_DQN.load_state_dict(torch.load(f'env_cart.pt'))
            else:
                if env_type == "cartpole":
                    DQN.load_state_dict(torch.load(f'env_lunar.pt'))
                else:
                    DQN.load_state_dict(torch.load(f'env_cart.pt'))

            print("loaded DQN")
        except:
            print("no pretrained DQN available")
            pass

    q_learning = DQN_agent(env=wrapped_env,DQN=DQN,target_DQN=target_DQN,eps=eps,eps_decay=eps_decay,gamma=gamma,batch_size=batch_size)
    start_time = time.time()
    q_learning.learn_policy(n_frames=n_frames)
    print(f'Time for {n_frames}: {time.time() - start_time}')

    if save_DQN:
        torch.save(DQN.state_dict(), f'env_lunar.pt')

