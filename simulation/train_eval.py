from recsim.simulator import runner_lib
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def run_train_experiment(
    base_dir,
    create_agent_fn,
    create_env_fn,
    env_config,
    episode_log_file="",
    max_training_steps=50,
    num_iterations=10
):
    """
    Runs a training experiment using TrainRunner.

    :param base_dir: Base directory to store logs and checkpoints.
    :param create_agent_fn: Function to create an agent.
    :param create_env_fn: Function to create an environment.
    :param env_config: Dictionary containing environment configuration.
    :param episode_log_file: Path to the episode log file (leave empty for no logging).
    :param max_training_steps: Maximum number of training steps per iteration.
    :param num_iterations: Number of training iterations.
    """
    # Create the environment using the provided function and configuration
    env = create_env_fn(env_config)
    
    # Initialize the TrainRunner
    runner = runner_lib.TrainRunner(
        base_dir=base_dir,
        create_agent_fn=create_agent_fn,
        env=env,
        episode_log_file=episode_log_file,
        max_training_steps=max_training_steps,
        num_iterations=num_iterations
    )
    
    # Run the experiment
    print("Starting the training experiment...")
    runner.run_experiment()
    print("Training experiment completed.")

def run_eval_experiment(
    base_dir,
    create_agent_fn,
    create_env_fn,
    env_config,
    max_eval_episodes=5,
    test_mode=True
):
    """
    Runs an evaluation experiment using EvalRunner.

    :param base_dir: Base directory to store logs and checkpoints.
    :param create_agent_fn: Function to create an agent.
    :param create_env_fn: Function to create an environment.
    :param env_config: Dictionary containing environment configuration.
    :param max_eval_episodes: Number of evaluation episodes to run.
    :param test_mode: Whether to run in test mode (agent uses greedy policy).
    """
    # Create the environment using the provided function and configuration
    env = create_env_fn(env_config)
    
    # Initialize the EvalRunner
    runner = runner_lib.EvalRunner(
        base_dir=base_dir,
        create_agent_fn=create_agent_fn,
        env=env,
        max_eval_episodes=max_eval_episodes,
        test_mode=test_mode
    )
    
    # Run the experiment
    print("Starting the evaluation experiment...")
    runner.run_experiment()
    print("Evaluation experiment completed.")