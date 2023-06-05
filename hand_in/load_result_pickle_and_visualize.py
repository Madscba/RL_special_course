import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = r'results\final\REINFORCE\seed3__env_Lunar__log_defau__fram10000__visu0__n_st40000__n_en1__algoREINF__gamm0.99__hidd256__lr0.001__eps1.0__eps_0.000__min_0.05__batc128__use_0__grad0__entr0__alph1__tau0.01__rewa2.0.pickle'
file_path = os.path.join(os.getcwd(),file_name)
# Load pickle file
with open(file_path, 'rb') as f:
    log_data = pickle.load(f)

class Plotter:
    def __init__(self, log_data):
        self.run_config = log_data["run_config"]
        self.episode_counter = log_data["Episodes"]
        self.episode_rewards = log_data["Episode Reward"]
        self.episode_lengths = log_data["Episode Length"]
        # self.exploration_rates = log_data["Episode Exploration Rate"]
        self.step_counter = log_data["Step"]
        self.step_rewards = log_data["Step Reward"]
        self.run_name = ""  # Provide a valid run name
        self.frame_interval = 10000  # Adjust frame interval if needed

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
        avg_rewards = [np.mean(interval) for interval in rew_intervals]
        intervals = [
            "{}:{}".format(i * self.frame_interval, (i + 1) * self.frame_interval)
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

# Create an instance of the Plotter class with the loaded log_data
plotter = Plotter(log_data)

# Call the plotting functions
plotter.plot_epi_rewards()
plotter.plot_step_rewards()
