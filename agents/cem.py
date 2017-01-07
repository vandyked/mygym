from agent_interface import AgentInterface
import numpy as np
from utils.constants import AGENT
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CEMAgent(AgentInterface):
    """
    Black box cross-entropy optimiser.
    Algorithm:
        - parametric model f(state, theta) = action
        - have a distribution over theta
        - REPEAT until solution found:
        -   sample a population of theta samples
        -   evaluate each
        -   recalculate theta distribution based on the top % of samples

    Notes:
        - MountainCar-v0 raised some issues. Is a lot of exploring until you get any kind of reward - here meaning
        episode stops before max of 200 steps.
    """
    def __init__(self, env, config):
        """
        :param env:
        :param config:
        """
        super(CEMAgent, self).__init__(env, config=config)
        self.population_size = config.getint(AGENT, "popsize")
        self.select_p = config.getfloat(AGENT, "p")
        self.sigma = config.getfloat(AGENT, "sigma")
        self.temperature = config.getfloat(AGENT, "smtemp")
        # model softmax(Ax + b)
        self.num_params = (self.state_space_dim + 1) * self.action_space_dim
        if self.train:
            self.mean_params = np.zeros((self.num_params,))
            self._reset_parameter_population()
            self._reset_tracking_stats()
            self.generation = 0
        else:
            param_dict = self.load_agent()
            self.mean_params = np.asarray(param_dict["mean"])
            self.episode_params = self._param_vec_to_dict(self.mean_params)

    def start_episode(self):
        if self.train:
            vec = self.params[self.episode_param_index, :]
            self.episode_params = self._param_vec_to_dict(vec)

    def act(self, ob, reward, done):
        logits = np.inner(self.episode_params["A"], ob) + self.episode_params["b"]
        return np.argmax(logits)
        # normalising is not necessary unless sampling
        #action_distribution = self.softmax(logits)
        #return np.argmax(action_distribution)

    def softmax(self, v):
        v = v/self.temperature
        ev = np.exp(v)
        dist = ev / np.sum(ev)
        return dist

    def end_episode(self, **kwargs):
        if self.train:
            self.param_evaluations[self.episode_param_index] = kwargs["total_reward"]
            self.episode_param_index += 1
            if self.episode_param_index == self.population_size:
                self._learning_step()

    def _reset_tracking_stats(self):
        self.param_evaluations = np.zeros(self.population_size)
        self.param_evaluations.fill(np.nan)
        self.episode_param_index = 0

    def _reset_parameter_population(self):
        self.params = np.random.multivariate_normal(mean=self.mean_params,
                                                    cov=self.sigma * np.identity(self.num_params),
                                                    size=(self.population_size,))

    def _param_vec_to_dict(self, vec):
        params = {}
        params["b"] = vec[:self.action_space.n]
        params["A"] = np.reshape(vec[self.action_space.n:], (self.action_space.n, self.state_space_dim))
        return params

    def _recalculate_mean(self):
        # recalculate mean
        sorted_indices = self.param_evaluations.argsort()[::-1]
        # take top p percent only
        selected_indices = sorted_indices[:np.floor(self.population_size * self.select_p)]
        self.mean_params = np.mean(self.params[selected_indices, :], axis=0)

    def _learning_step(self):
        # output
        logger.info("Generation: {} Mean reward: {}".format(self.generation, np.mean(self.param_evaluations)))
        self._recalculate_mean()
        self._reset_parameter_population()
        # reset state stats:
        self._reset_tracking_stats()
        self.generation += 1

    def _get_save_dict(self):
        return {"mean": self.mean_params.tolist()}

