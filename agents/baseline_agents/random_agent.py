from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl import logging
from recsim import agent


class RandomAgent(agent.AbstractEpisodicRecommenderAgent):
    """An agent that recommends a random slate of documents."""

    def __init__(self, action_space, random_seed=0):
        """
        Initializes the RandomAgent.

        Args:
            action_space: The action space defining the possible actions.
            slate_size (int): The number of documents to recommend in each slate.
            random_seed (int): Seed for reproducibility of random actions.
        """
        super(RandomAgent, self).__init__(action_space)
        np.random.seed(random_seed)    # Set the global random seed for reproducibility

    def step(self, reward, observation):
        """
        Records the most recent transition and returns the agent's next action.

        Args:
            reward: Unused (random agent does not learn from rewards).
            observation: A dictionary that includes the most recent observation.
                         Should include 'doc' field with observations of all candidates.

        Returns:
            slate (list): A list of document indices representing the recommended slate.
        """
        del reward  # Unused argument
        doc_obs = observation['doc']

        # Generate random slate
        doc_ids = list(range(len(doc_obs)))
        np.random.shuffle(doc_ids)  # Direct use of numpy's random shuffle

        # Select top documents after shuffling
        slate = doc_ids[:self._slate_size]
        logging.debug(f'Recommended slate: {slate}')

        return slate

def create_random_agent(environment):
  action_space = environment.action_space
  return RandomAgent(action_space, random_seed=0)