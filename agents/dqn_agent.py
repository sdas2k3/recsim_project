# from recsim.agents import full_slate_q_agent
from recsim.agents import tabular_q_agent
import tensorflow.compat.v1 as tf

# def create_fullstate_agent(sess, environment, eval_mode, summary_writer=None):
#   kwargs = {
#       'observation_space': environment.observation_space,
#       'action_space': environment.action_space,
#       'summary_writer': summary_writer,
#       'eval_mode': eval_mode,
#   }
#   return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)

# def create_dqn_agent(environment):
#   sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#   return create_fullstate_agent(sess,environment,eval_mode=False)

def create_tabular_q_agent(environment):
    action_space = environment.action_space
    observation_space = environment.observation_space
    return tabular_q_agent.TabularQAgent(observation_space,action_space,eval_mode=False)
