import numpy as np
from environment.movielens_env import create_environment
from agents.dqn_agent import create_tabular_q_agent
from agents.baseline_agents.random_agent import create_random_agent
from agents.baseline_agents.greedy_max_agent import create_greedy_max_agent
from agents.baseline_agents.greedy_min_agent import create_greedy_min_agent
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

def run_simulation(seed=0, num_train_episodes=30, num_eval_episodes=100, num_iterations=10, max_steps_per_episode=500):
    """
    Runs the simulation by training and evaluating different agents.

    Args:
        seed (int): Random seed for reproducibility.
        num_train_episodes (int): Number of training episodes.
        num_eval_episodes (int): Number of evaluation episodes.
        num_iterations (int): Number of training iterations per episode.
        max_steps_per_episode (int): Maximum steps per episode.

    Returns:
        dict: A dictionary containing training and evaluation results for each agent.
    """
    np.random.seed(seed)

    env_config = {
        'num_candidates': 20,
        'slate_size': 1,
        'resample_documents': True,
        'seed': seed,
    }
    
    environment = create_environment(env_config)

    # Initialize agents
    tabular_q_agent = create_tabular_q_agent(environment)
    random_agent = create_random_agent(environment)
    greedy_max_agent = create_greedy_max_agent(environment)
    greedy_min_agent = create_greedy_min_agent(environment)

    # Define agent configurations
    agent_configs = [
        {"name": "Tabular Q", "agent": tabular_q_agent},
        {"name": "Random", "agent": random_agent},
        {"name": "Greedy Max", "agent": greedy_max_agent},
        {"name": "Greedy Min", "agent": greedy_min_agent}
    ]

    results = {}

    for config in agent_configs:
        agent_name = config["name"]
        agent = config["agent"]

        # Train agent
        trainer = Trainer(
            agent=agent,
            env=create_environment(env_config),
            logs_dir=f"./logs/training_logs/{agent_name}",
            num_episodes=num_train_episodes,
            num_iterations=num_iterations,
            max_steps_per_episode=max_steps_per_episode,
            agent_name=agent_name
        )

        train_episode_rewards, train_avg_reward = trainer.train()

        # Evaluate agent
        evaluator = Evaluator(
            agent=agent,
            env=create_environment(env_config),
            logs_dir=f"./logs/evaluation_logs/{agent_name}",
            num_episodes=num_eval_episodes,
            max_steps_per_episode=max_steps_per_episode,
            agent_name=agent_name
        )

        eval_episode_rewards, eval_avg_reward = evaluator.evaluate()

        # Store results
        results[agent_name] = {
            "train_results": {
                "episode_rewards": train_episode_rewards,
                "avg_reward": train_avg_reward
            },
            "eval_results": {
                "episode_rewards": eval_episode_rewards,
                "avg_reward": eval_avg_reward
            }
        }

    return results
