import gymnasium as gym
import numpy as np


class CartpoleEnv():
   def __init__(self,render:bool = True):
      if render:
         self.env = gym.make('CartPole-v1',render_mode="human")
      else:
         self.env = gym.make('CartPole-v1')
      self.observation, self.info = self.env.reset(seed=42)


   def run_env_w_random_actions(self,n_steps:int=100):
      for _ in range(n_steps):
         action = self.env.action_space.sample()  # this is where you would insert your policy
         observation, reward, terminated, truncated, info = self.env.step(action)
         if terminated or truncated:
            self.observation, self.info = self.env.reset()
      self.env.close()



if __name__ == "__main__":
   env = CartpoleEnv()
   env.run_env_w_random_actions()