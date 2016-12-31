from agents.agent_interface import AgentInterface
from agents.cem import CEMAgent
from agents.policygradient import PolicyGradientAgent
from agents.dqn import DQNAgent


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_agent(agent_name, env, config):
    if agent_name == AgentInterface.__name__:
        logger.debug("Loading AgentInterface/RandomAgent")
        return AgentInterface(env, config=config)
    elif agent_name == CEMAgent.__name__:
        logger.debug("Loading black box Cross-Entropy Agent")
        return CEMAgent(env, config=config)
    elif agent_name == PolicyGradientAgent.__name__:
        logger.debug("Loading PolicyGradient Agent")
        return PolicyGradientAgent(env, config=config)
    elif agent_name == DQNAgent.__name__:
        logger.debug("Loading Deep Q-leaning Agent")
        return DQNAgent(env, config=config)
    else:
        logger.error("No agent named: {}".format(agent_name))
        exit()
