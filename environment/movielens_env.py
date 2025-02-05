from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
import gin.tf
import gym
from gym import spaces
import numpy as np
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym

import numpy as np
from scipy import stats
import random
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

# Genre clickbait scores as a constant dictionary
genre_clickbait_scores = {
    'Action': 1, 'Adventure': 1, 'Animation': -1, 'Children\'s': -1, 'Comedy': 1,
    'Crime': 1, 'Documentary': -1, 'Drama': -1, 'Fantasy': -1, 'Film-Noir': -1,
    'Horror': 1, 'Musical': -1, 'Mystery': 1, 'Romance': -1, 'Sci-Fi': -1,
    'Thriller': 1, 'War': 1, 'Western': -1
}

class LTSDocument(document.AbstractDocument):
    """Document class representing LTS with a clickbait score and genre vector."""

    def __init__(self, doc_id, title, genre_vector):
        self.genre_vector = genre_vector
        self._doc_id = doc_id
        self.title = title
        self.clickbait_score = self.calculate_clickbait_score(genre_vector)
        super(LTSDocument, self).__init__(doc_id)

    def calculate_clickbait_score(self, genre_vector):
        """Calculate clickbait score based on genre vector and weights."""
        kaley_weight = 0
        chocolatey_weight = 0

        for i, genre in enumerate(genre_clickbait_scores):
            if genre_vector[i] == 1:
                score = genre_clickbait_scores[genre]
                if score > 0:  # Chocolatey score
                    chocolatey_weight += score
                else:  # Kaley score
                    kaley_weight += abs(score)

        total_weight = kaley_weight + chocolatey_weight

        # Handle cases where no genres contribute to the score
        if total_weight == 0:
            return 0.5  # Neutral score when no genres are present

        # Generate score in either clickbait (>0.8) or non-clickbait (<0.2) range
        if chocolatey_weight > kaley_weight:
            return 0.8 + 0.2 * random.uniform(0, 1)
        else:
            return 0.2 * random.uniform(0, 1)

    def create_observation(self):
        """Create an observation combining clickbait score and genre vector."""
        return np.array([self.clickbait_score, self._doc_id])

    @staticmethod
    def observation_space():
        """Define observation space for the document."""
        return spaces.Box(shape=(2,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return f"Document {self._doc_id} with clickbait_score {self.clickbait_score:.2f}."


class LTSDocumentSampler(document.AbstractDocumentSampler):
    """Sampler class to generate LTSDocuments from a movie dataset."""

    def __init__(self, doc_ctor=LTSDocument, **kwargs):
        super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self.movies = pd.read_csv('./data/processed/movielens_processed/movies.csv')

    def sample_document(self):
        """Sample a movie and create an LTSDocument."""
        movie = self.movies.iloc[self._doc_count]
        genre_vector = movie[list(genre_clickbait_scores.keys())].values.astype(np.float32)
        doc_features = {
            'doc_id': movie['movie_id'],
            'title': movie['title'],
            'genre_vector': genre_vector
        }
        self._doc_count = (self._doc_count + 1) % len(self.movies)
        return self._doc_ctor(**doc_features)

    @staticmethod
    def observation_space():
        """Define observation space for the sampler."""
        return spaces.Box(shape=(19,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return f"LTSDocumentSampler with {len(self.movies)} movies."


class LTSUserState(user.AbstractUserState):
    """Class to represent users with attributes from ML-1M dataset."""

    def __init__(self, memory_discount, sensitivity, innovation_stddev,
                 choc_mean, choc_stddev, kale_mean, kale_stddev,
                 net_positive_exposure, time_budget, gender, age, user_id):
        super().__init__()
        self.memory_discount = memory_discount
        self.sensitivity = sensitivity
        self.innovation_stddev = innovation_stddev

        self.choc_mean = choc_mean
        self.choc_stddev = choc_stddev
        self.kale_mean = kale_mean
        self.kale_stddev = kale_stddev

        self.net_positive_exposure = net_positive_exposure
        self.satisfaction = 1 / (1 + np.exp(-sensitivity * net_positive_exposure))
        self.time_budget = time_budget

        # User attributes from users.dat
        self.gender = gender
        self.age = age
        self._user_id = user_id

    def score_document(self, doc_obs):
        return 1

    def create_observation(self):
        """User's state is not observable, but attributes are."""
        return np.array([self._user_id, self.gender, self.age])

    @staticmethod
    def observation_space():
        """Define observation space for user attributes."""
        return spaces.Box(shape=(3,), dtype=np.float32, low=0.0, high=np.inf)


class LTSUserSampler(user.AbstractUserSampler):
    """Generates user states with attributes from ML-1M dataset."""

    def __init__(self, 
                 user_ctor=LTSUserState, 
                #  memory_discount=0.7,
                #  sensitivity=0.01,
                #  innovation_stddev=0.05,
                #  choc_mean=5.0,
                #  choc_stddev=1.0,
                #  kale_mean=4.0,
                #  kale_stddev=1.0,
                #  time_budget=10,
                memory_discount=0.7,
                sensitivity=0.9,
                innovation_stddev=1e-5,
                choc_mean=5.0,
                choc_stddev=0.0,
                kale_mean=0.0,
                kale_stddev=0.0,
                time_budget=10,
                 **kwargs):
        super().__init__(user_ctor, **kwargs)
        
        # Default state parameters
        self._state_parameters = {
            'memory_discount': memory_discount,
            'sensitivity': sensitivity,
            'innovation_stddev': innovation_stddev,
            'choc_mean': choc_mean,
            'choc_stddev': choc_stddev,
            'kale_mean': kale_mean,
            'kale_stddev': kale_stddev,
            'time_budget': time_budget
        }

        # Load users.dat data
        self.users = pd.read_csv('./data/processed/movielens_processed/users.csv')
        self.users['gender'] = self.users['gender'].map({'M': 1, 'F': 0})
        self._user_index = 0

    def sample_user(self):
        """Sample a user from the users dataset."""
        user_row = self.users.iloc[self._user_index]
        self._user_index = (self._user_index + 1) % len(self.users)

        if(self._user_index >= len(self.users)):
          self.users = self.users.sample(frac=1).reset_index(drop=True)
          self._user_index = 0

        # Extract user attributes
        gender = user_row['gender']
        age = user_row['age']
        # occupation = user_row['occupation']

        # Add user-specific attributes to state parameters
        state_parameters = self._state_parameters.copy()
        state_parameters.update({
            'net_positive_exposure': 1,
            # ((self._rng.random_sample() - 0.5) *
            #                           (1 / (1.0 - state_parameters['memory_discount']))),
            'gender': gender,
            'age': age,
            # 'occupation': occupation,
            'user_id' : user_row['user_id']
        })

        return self._user_ctor(**state_parameters)

    def __str__(self):
        return f"LTSStaticUserSampler with {len(self.users)} users."

class LTSResponse(user.AbstractResponse):
    MAX_ENGAGEMENT_MAGNITUDE = 100.0

    def __init__(self, clicked=False, engagement=0.0):
        self.clicked = clicked
        self.engagement = engagement

    def create_observation(self):
        return {'click': int(self.clicked), 'engagement': np.array(self.engagement)}

    @classmethod
    def response_space(cls):
        return spaces.Dict({
            'click': spaces.Discrete(2),
            'engagement': spaces.Box(
                low=0.0,
                high=cls.MAX_ENGAGEMENT_MAGNITUDE,
                shape=tuple(),
                dtype=np.float32)
        })

class LTSUserModel(user.AbstractUserModel):
  """Class to model a user with long-term satisfaction dynamics.

  Implements a controlled continuous Hidden Markov Model of the user having
  the following components.
    * State space: one dimensional real number, termed net_positive_exposure
      (abbreviated NPE);
    * controls: one dimensional control signal in [0, 1], representing the
      clickbait score of the item of content;
    * transition dynamics: net_positive_exposure is updated according to:
      NPE_(t+1) := memory_discount * NPE_t
                   + 2 * (clickbait_score - .5)
                   + N(0, innovation_stddev);
    * observation space: a nonnegative real number, representing the degree of
      engagement, e.g. econds watched from a recommended video. An observation
      is drawn from a log-normal distribution with mean

      (clickbait_score * choc_mean
                      + (1 - clickbait_score) * kale_mean) * SAT_t,

      where SAT_t = sigmoid(sensitivity * NPE_t). The observation standard
      standard deviation is similarly given by

      (clickbait_score * choc_stddev + ((1 - clickbait_score) * kale_stddev)).

      An individual user is thus represented by the combination of parameters
      (memory_discount, innovation_stddev, choc_mean, choc_stddev, kale_mean,
      kale_stddev, sensitivity), which are encapsulated in LTSUserState.

    Args:
      slate_size: An integer representing the size of the slate
      user_state_ctor: A constructor to create user state.
      response_model_ctor: A constructor function to create response. The
        function should take a string of doc ID as input and returns a
        LTSResponse object.
      seed: an integer as the seed in random sampling.
  """

  def __init__(self,
               slate_size,
               user_state_ctor=None,
               response_model_ctor=None,
               seed=0):
    if not response_model_ctor:
      raise TypeError('response_model_ctor is a required callable.')

    super(LTSUserModel, self).__init__(
        response_model_ctor,
        LTSUserSampler(user_ctor=user_state_ctor, seed=seed), slate_size)

  def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return self._user_state.time_budget <= 0

  def update_state(self, slate_documents, responses):
    """Updates the user's latent state based on responses to the slate.

    Args:
      slate_documents: a list of LTSDocuments representing the slate
      responses: a list of LTSResponses representing the user's response to each
        document in the slate.
    """

    for doc, response in zip(slate_documents, responses):
      if response.clicked:
        innovation = np.random.normal(scale=self._user_state.innovation_stddev)
        net_positive_exposure = (self._user_state.memory_discount
                                 * self._user_state.net_positive_exposure
                                 - 2.0 * (doc.clickbait_score - 0.5)
                                 + innovation
                                )
        self._user_state.net_positive_exposure = net_positive_exposure
        satisfaction = 1 / (1.0 + np.exp(-self._user_state.sensitivity
                                         * net_positive_exposure)
                           )
        self._user_state.satisfaction = satisfaction
        self._user_state.time_budget -= 1
        return

  def simulate_response(self, documents):
    """Simulates the user's response to a slate of documents with choice model.

    Args:
      documents: a list of LTSDocument objects.

    Returns:
      responses: a list of LTSResponse objects, one for each document.
    """
    # List of empty responses
    responses = [self._response_model_ctor() for _ in documents]
    # User always clicks the first item.
    selected_index = 0
    self.generate_response(documents[selected_index], responses[selected_index])
    return responses

  def generate_response(self, doc, response):
    """Generates a response to a clicked document.

    Args:
      doc: an LTSDocument object.
      response: an LTSResponse for the document.
    Updates: response, with whether the document was clicked, liked, and how
      much of it was watched.
    """
    response.clicked = True
    # linear interpolation between choc and kale.
    engagement_loc = (doc.clickbait_score * self._user_state.choc_mean
                      + (1 - doc.clickbait_score) * self._user_state.kale_mean)
    engagement_loc *= self._user_state.satisfaction
    engagement_scale = (doc.clickbait_score * self._user_state.choc_stddev
                        + ((1 - doc.clickbait_score)
                           * self._user_state.kale_stddev))
    log_engagement = np.random.normal(loc=engagement_loc,
                                      scale=engagement_scale)
    response.engagement = np.exp(log_engagement)

def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
      if response.clicked:
        reward += response.engagement
    return reward

def create_environment(env_config):
  """Creates a long-term satisfaction environment."""

  user_model = LTSUserModel(
      env_config['slate_size'],
      user_state_ctor=LTSUserState,
      response_model_ctor=LTSResponse)

  document_sampler = LTSDocumentSampler()

  ltsenv = environment.Environment(
      user_model,
      document_sampler,
      env_config['num_candidates'],
      env_config['slate_size'],
      resample_documents=env_config['resample_documents'])

  return recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)