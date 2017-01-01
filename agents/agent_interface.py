import ConfigParser
from utils.constants import AGENT
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AgentInterface(object):
    def __init__(self, env, config=None):
        self.action_space = env.action_space
        self.action_space_dim = self.action_space.n
        try:
            if hasattr(env, 'state'):
                self.state_space_shape = env.state.shape
            elif hasattr(env, 'observation_space'):
                self.state_space_shape = env.observation_space.shape
        except AttributeError:
            exit("Failed to get state dimension from environment")
        # dimension of state space:
        self.state_space_dim = 1
        for i in self.state_space_shape:
            self.state_space_dim *= i

        # optional arguments that are shared by some agents only:
        self.epsilon = 1.0
        self.epsilon_limit = 0.1
        self.epsilon_decay_rate = 0.1
        self.train = True
        try:
            self.epsilon = config.getfloat(AGENT, "epsilon")
            self.epsilon_decay_rate = config.getfloat(AGENT, "epsilondecayrate")
            self.train = config.getboolean(AGENT, "train")
        except ConfigParser.NoOptionError:
            logger.warning("Using some default config options")

    def act(self, ob, reward, done):
        """ RandomAgent for interface
        :param ob:
        :param reward:
        :param done:
        :return:
        """
        return self.action_space.sample()

    def cleanup(self, **kwargs):
        pass

    def start_episode(self):
        pass

    def end_episode(self, **kwargs):
        pass