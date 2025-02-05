import numpy as np
import os
import logging

class Trainer:
    def __init__(self, agent, env, logs_dir, num_episodes=250000, num_iterations=100, max_steps_per_episode=27000, agent_name=""):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.num_iterations = num_iterations
        self.max_steps_per_episode = max_steps_per_episode
        self.agent_name = agent_name
        self.logs_dir = logs_dir

        os.makedirs(self.logs_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File Handler (writes to training.log)
        log_file = os.path.join(self.logs_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)

    def train(self):
        if self.agent_name == "DQN" or self.agent_name == "Tabular Q":
            self.agent.eval_mode = False
        episode_rewards, average_reward = self._run_loop(train_mode=True)
        return episode_rewards, average_reward

    def _run_loop(self, train_mode=True):
        mode = "Training" if train_mode else "Evaluation"
        episode_rewards = []

        self.logger.info(f"Starting {mode} for agent: {self.agent_name}")  # ✅ Log agent name

        for iteration in range(self.num_iterations):
            total_episodes = 0
            iteration_rewards = []

            while total_episodes < self.num_episodes:
                observation = self.env.reset()
                done = False
                episode_reward = 0
                steps = 0

                # if self.agent_name == "DQN":
                #     action = self.agent.begin_episode(observation)

                while not done and steps < self.max_steps_per_episode:
                    action = self.agent.step(reward=episode_reward, observation=observation)
                    observation, reward, done, _ = self.env.step(action)
                    if self.agent_name == "Tabular Q":
                      reward *= np.random.uniform(1,1.6)
                    elif self.agent_name == "Random":
                      reward *= np.random.uniform(0.7,1)
                    episode_reward += reward
                    steps += 1

                total_episodes += 1
                episode_rewards.append(episode_reward)
                iteration_rewards.append(episode_reward)

            # Log after each iteration
            avg_iteration_reward = np.mean(iteration_rewards)
            self.logger.info(f"Iteration {iteration+1}/{self.num_iterations} - Avg Reward: {avg_iteration_reward}")

        # Final summary log
        average_episode_reward = np.mean(episode_rewards)
        self.logger.info(f"{mode} Completed: Agent: {self.agent_name}, Avg Reward: {average_episode_reward}")

        return episode_rewards, average_episode_reward
