import os
import matplotlib.pyplot as plt
import numpy as np
class Visualizer:
    def __init__(self, results, save_dir="./results/plots"):
        self.results = results
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_episode_rewards(self):
        """Plots episode-wise rewards separately for Training and Evaluation."""
      
        # ----- Plot for Training Rewards -----
        plt.figure(figsize=(12, 6))
        for agent_name, data in self.results.items():
            train_rewards = data["train_results"]["episode_rewards"]
            plt.plot(train_rewards, label=f"{agent_name}", linestyle="-")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards (Training)")
        plt.legend()
        plt.grid(True)

        # Save Training Plot
        train_save_path = os.path.join(self.save_dir, "training_episode_rewards.png")
        plt.savefig(train_save_path)
        print(f"Saved: {train_save_path}")
        plt.show()

        # ----- Plot for Evaluation Rewards -----
        plt.figure(figsize=(12, 6))
        for agent_name, data in self.results.items():
            eval_rewards = data["eval_results"]["episode_rewards"]
            plt.plot(eval_rewards, label=f"{agent_name}", linestyle="--")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards (Evaluation)")
        plt.legend()
        plt.grid(True)

        # Save Evaluation Plot
        eval_save_path = os.path.join(self.save_dir, "evaluation_episode_rewards.png")
        plt.savefig(eval_save_path)
        print(f"Saved: {eval_save_path}")
        plt.show()


    def plot_avg_rewards(self):
        """Plots a bar chart comparing average rewards of agents."""
        agent_names = []
        train_avg_rewards = []
        eval_avg_rewards = []

        for agent_name, data in self.results.items():
            agent_names.append(agent_name)
            train_avg_rewards.append(data["train_results"]["avg_reward"])
            eval_avg_rewards.append(data["eval_results"]["avg_reward"])

        x = np.arange(len(agent_names))

        plt.figure(figsize=(10, 6))
        plt.bar(x - 0.2, train_avg_rewards, width=0.4, label="Training", color="blue")
        plt.bar(x + 0.2, eval_avg_rewards, width=0.4, label="Evaluation", color="orange")

        plt.xticks(ticks=x, labels=agent_names)
        plt.ylabel("Average Reward")
        plt.title("Comparison of Average Rewards (Train vs Eval)")
        plt.legend()
        plt.grid(axis="y")

        save_path = os.path.join(self.save_dir, "average_rewards.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.show()

    def generate_all_plots(self):
        """Runs all visualization functions."""
        self.plot_episode_rewards()
        self.plot_avg_rewards()