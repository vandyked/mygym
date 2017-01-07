import ConfigParser
from utils.constants import ENV, TRAINER
from utils.constants import AGENT
from utils.constants import MODEL_TRAINED
from utils.constants import JSON
from utils.constants import INI
import json
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AgentInterface(object):
    def __init__(self, env, config):
        self.config = config
        self.action_space = env.action_space
        self.action_space_dim = self.action_space.n
        self.model_name = self._get_model_name(config)
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

        # Optional arguments that are shared by some agents only:
        # -- These should effect learning only, such that a config and saved model should
        # -- be enough to load and run a trained model
        self.epsilon_limit = 0.1
        try:
            self.epsilon = config.getfloat(AGENT, "epsilon")
        except:
            self.epsilon = 0.5
        try:
            self.epsilon_decay_rate = 0.9
        except:
            self.epsilon_decay_rate = config.getfloat(AGENT, "epsilondecayrate")
        try:
            self.train = config.getboolean(AGENT, "train")
        except:
            self.train = True

    def act(self, ob, reward, done):
        """ RandomAgent for interface
        :param ob:
        :param reward:
        :param done:
        :return:
        """
        return self.action_space.sample()

    def cleanup(self, **kwargs):
        if self.train:
            self._save_agent()
            self._save_config()

    def load_agent(self):
        load_name = self.config.get(AGENT, "loadname")
        with open(load_name, 'r') as datafile:
            return json.load(datafile)

    def start_episode(self):
        pass

    def end_episode(self, **kwargs):
        pass

    def _save_agent(self):
        """ Override this if model doesn't suit being dumped to json. Must set self.load_name
        """
        if not self.train:
            return
        self.load_name = os.path.join(MODEL_TRAINED, self.model_name + JSON)
        with open(self.load_name, 'w') as datafile:
            json.dump(self._get_save_dict(), datafile)

    def _save_config(self):
        self.config.set(AGENT, "loadname", self.load_name)
        self.config.set(AGENT, "train", False)
        self.config.set(AGENT, "epsilon", 0.0)
        self.config.set(TRAINER, "render", True)
        self.config.set(TRAINER, "iterations", 10)
        config_name = os.path.join(MODEL_TRAINED, self.model_name + INI)
        with open(config_name, 'w') as cfg_file:
            self.config.write(cfg_file)

    @staticmethod
    def _get_model_name(config):
        env_name = config.get(ENV, "id")
        try:
            approximator_name = config.get(AGENT, "approximator")
        except ConfigParser.NoOptionError:
            approximator_name = "na"
        agent_name = config.get(AGENT, "id")
        return '_'.join([env_name, agent_name, approximator_name])

    def _get_save_dict(self):
        pass
