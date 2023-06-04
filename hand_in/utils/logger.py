# A module for logging and monitoring training progress.

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class RLLogger:
    def __init__(
        self,
        argparser,
        frame_interval: int = 10000,
        log_format: str = "console",
        log_frequency: int = 1,
    ):
        self.frame_interval = argparser.args.frame_interval
        self.log_frequency = log_frequency
        self.log_format = argparser.args.log_format
        self.run_config = vars(argparser.args)
        self.run_name = "__".join(
            [key[:4] + "" + str(val)[:5] for key, val in self.run_config.items()]
        )
        self.step_rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        # self.exploration_rates = []
        self.episode_counter = 0
        self.step_counter = 0
        self.current_episode_frame_counter = 0
        self.current_episode_reward = 0

    def log_episode(self):
        self.episode_counter += 1
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_frame_counter)
        # self.exploration_rates.append(exploration_rate)

        if self.episode_counter % self.log_frequency == 0:
            if self.log_format == "console":
                self._print_epi_log()
            elif self.log_format == "file":
                self._write_to_file()
            else:
                self._print_epi_log()
                self._write_to_file()

        self.reset_epi_stats()

    def log_step(self, reward):
        self.step_counter += 1
        self.step_rewards.append(reward[0])
        self.current_episode_reward += reward[0]
        self.current_episode_frame_counter += 1

    def reset_epi_stats(self):
        self.current_episode_reward = 0
        self.current_episode_frame_counter = 0

    def _print_epi_log(self):
        reward = self.episode_rewards[self.episode_counter - 1]
        length = self.episode_lengths[self.episode_counter - 1]
        # exp_rate = self.exploration_rates[self.episode_counter-1]
        print(
            f"\nEpisode {self.episode_counter}: Reward = {reward =:.2f}, Length = {length:.2f}"
        )

        print(f"Avg rew 50epi: {np.mean(self.episode_rewards[:self.episode_counter - 1][-50:])}")


    def _write_to_file(self):
        log_data = {
            "run_config": self.run_config,
            "Episodes": self.episode_counter,
            "Episode Reward": self.episode_rewards,
            "Episode Length": self.episode_lengths,
            # 'Episode Exploration Rate': self.exploration_rates,
            "Step": self.step_counter,
            "Step Reward": self.step_rewards,
        }
        with open(
            f"{os.path.join(os.getcwd(), 'results', 'temporary', f'{self.run_name}.pickle')}",
            "wb",
        ) as outfile:
            pickle.dump(log_data, outfile)
        #print(
        #    f"Saving log to file: {os.path.join(os.getcwd(), 'results', 'temporary', f'{self.run_name}.pickle')}"
        #)

    def plot_epi_rewards(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Progress")
        plt.savefig(f'epi_rew_{self.run_name}.png')
        plt.show()

        plt.plot(self.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Length of episodes")
        plt.savefig(f'epi_length_{self.run_name}.png')
        plt.show()

    def plot_step_rewards(self):
        print("Check that intervals are found correctly")
        rew_intervals = np.array_split(
            np.array(self.step_rewards),
            self.run_config["n_steps"] // self.frame_interval,
        )
        # rew_intervals = np.split(np.array(self.step_rewards), self.run_name['n_steps']//self.frame_interval, axis=1)
        avg_rewards = [np.mean(interval) for interval in rew_intervals]
        intervals = [
            "{}:{}".format(i * (self.frame_interval), (i + 1) * self.frame_interval)
            for i in range(self.run_config["n_steps"] // self.frame_interval)
        ]
        plt.xticks(
            np.linspace(0, self.run_config["n_steps"] // self.frame_interval, 10),
            intervals[:: (self.run_config["n_steps"] // self.frame_interval // 10)],
            rotation=60,
        )
        plt.plot(avg_rewards)
        plt.subplots_adjust(bottom=0.3)
        plt.xlabel("Steps")
        plt.ylabel("Average Reward pr. {} frames".format(self.frame_interval))
        plt.title(f"Reward Progress \n{self.run_name}")
        x_max = np.max(avg_rewards)
        y_max = np.where(avg_rewards == x_max)[0][0]
        plt.scatter(
            np.where(avg_rewards == x_max)[0][0],
            x_max,
            label="maximum avg. rew: {:.2f}".format(x_max),
        )
        plt.legend(loc="lower right")
        plt.savefig(f'avg_step_rew_{self.run_name}.png')
        plt.show()


def get_logger(argparser):
    logger = RLLogger(argparser)
    return logger
