from agent_interface import AgentInterface
from utils.load_approximators import load_approximator
from utils.constants import AGENT
import numpy as np


class PolicyGradientAgent(AgentInterface):
    """
    Algorithm:

    """
    def __init__(self, env, config):
        super(PolicyGradientAgent, self).__init__(env, config=config)
        approximator_name = config.get(AGENT, "approximator")

        self.approximator = load_approximator(approximator_name,
                                              config=config,
                                              inputDim=self.state_space_dim,
                                              outputDim=self.action_space_dim)

        # TODO  implement policy gradient training

    def act(self, ob, reward, done):
        action_distribution = self.approximator.predict(ob)
        return np.argmax(action_distribution)
