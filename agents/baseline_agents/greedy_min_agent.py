import numpy as np
from absl import logging
from recsim import agent
from recsim import choice_model as cm

class GreedyMinAgent(agent.AbstractEpisodicRecommenderAgent):
    """An agent that recommends slates with the lowest values (minimizing) based on the first element of document features."""

    def __init__(self,
                 action_space,
                 choice_model=cm.MultinomialLogitChoiceModel({'no_click_mass': 5})):
        """Initializes a new greedy min agent.

        Args:
            action_space: A gym.spaces object that specifies the format of actions.
            belief_state: An instantiation of AbstractUserState assumed by the agent.
            choice_model: An instantiation of AbstractChoiceModel assumed by the agent. Default to multinomial logit choice model with no_click_mass = 5.
        """
        super(GreedyMinAgent, self).__init__(action_space)
        self._choice_model = choice_model

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        Args:
            reward: Unused.
            observation: A dictionary that includes the most recent observations and should have the following fields:
                - user: A list of floats representing the user's observed state.
                - doc: A list of observations of document features.

        Returns:
            slate: An integer array of size _slate_size, where each element is an index into the list of doc_obs.
        """
        del reward  # Unused argument.
        doc_obs = observation['doc']

        # Find the indices of the lowest scoring documents
        slate = self.findBestDocuments(doc_obs)

        # Return the slate of those documents
        logging.debug('Recommended slate: %s', slate)
        return slate

    def findBestDocuments(self, doc_obs):
        """Returns the indices of the lowest scores in sorted order based on the first element of doc_obs values.

        Args:
            doc_obs: An OrderedDict where values are arrays. We use the first element of the array to sort.

        Returns:
            sorted_indices: A list of integers indexing the lowest scores, in sorted order based on the first element of doc.
        """

        # Extract the first element of each array in doc_obs and use it for sorting
        doc_first_elements = [doc[0] for doc in doc_obs.values()]

        # Choose the k = slate_size best ones based on the first element of doc
        indices = np.argpartition(doc_first_elements, self._slate_size)[:self._slate_size]

        # Sort them so the best appear first based on the first element of doc_obs
        sorted_indices = indices[np.argsort(np.array(doc_first_elements)[indices])]
        return sorted_indices

def create_greedy_min_agent(env, choice_model=None):
    """
    Creates an instance of the GreedyMinAgent.

    :param env: The environment object used to get the action space.
    :param choice_model: The choice model used by the agent (default: None).
    :return: An instance of GreedyMinAgent.
    """

    return GreedyMinAgent(action_space=env.action_space)
