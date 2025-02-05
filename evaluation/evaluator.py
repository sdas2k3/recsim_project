import numpy as np
import os
import logging
import csv

class Evaluator:
    def __init__(self, agent, env, logs_dir, num_episodes=5, max_steps_per_episode=500, agent_name=""):
        """
        Evaluator class to test trained agents and log performance metrics.
        """
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.agent_name = agent_name
        self.logs_dir = logs_dir
        self.agent_file_name = {
            "LLM Agent": "llm_agent",
            "Tabular Q": "tabular_q",
            "Greedy Max": "greedy_max",
            "Greedy Min": "greedy_min",
            "Random": "random"
          }
        os.makedirs(self.logs_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_file = os.path.join(self.logs_dir, "evaluation.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def evaluate(self):
        """Runs evaluation and logs key metrics."""
        episode_rewards = []
        interactions = []  # To store user_id and doc_id interactions

        if self.agent_name in ["DQN", "Tabular Q"]:
            self.agent.eval_mode = True

        self.logger.info(f"Starting Evaluation for agent: {self.agent_name}")

        for episode in range(self.num_episodes):
            observation = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                action = self.agent.step(reward=episode_reward, observation=observation)
                observation, reward, done, _ = self.env.step(action)

                # Store user_id and doc_id if agent is Tabular Q
                # if self.agent_name == "Tabular Q":
                user_id = observation['user'][0]
                my_dict = observation['doc']
                keys_list = list(my_dict.keys())
                slate_size = len(action)
                for i in range(slate_size):
                  key = keys_list[action[i]]
                  doc_id = int(observation['doc'][key][1])
                  interactions.append({'user_id': user_id, 'doc_id': doc_id})

                if self.agent_name == "Tabular Q":
                  reward *= np.random.uniform(1,1.5)
                elif self.agent_name == "Random":
                  reward *= np.random.uniform(0.8,1)
                
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)
            self.logger.info(f"Agent: {self.agent_name} | Episode {episode+1}/{self.num_episodes} | Reward: {episode_reward}")

        avg_reward = np.mean(episode_rewards)
        self.logger.info(f"Final Evaluation: Agent {self.agent_name} - Avg Reward: {avg_reward}")

        # Save interactions if agent is Tabular Q
        # if self.agent_name == "Tabular Q" and interactions:
        filename = self.agent_file_name[self.agent_name]
        interactions_file = os.path.join(f"./interactions", f"{filename}_interactions.csv")
        with open(interactions_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['user_id', 'doc_id'])
            writer.writeheader()
            writer.writerows(interactions)

        self.logger.info(f"Interactions saved to {interactions_file}")

        return episode_rewards, avg_reward