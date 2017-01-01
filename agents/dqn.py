from agent_interface import AgentInterface
from utils.load_approximators import load_approximator
from utils.constants import AGENT, HDF5, MODEL_CHECKPOINTS
import os
import numpy as np
from copy import copy
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ReplayMemory(object):
    def __init__(self, buffer_limit=1000):
        self.buffer = []
        self.buffer_limit = buffer_limit

    def is_buffer_full(self):
        return len(self.buffer) >= self.buffer_limit    # only start learning once buffer is full

    def sample(self, batch_size=16):
        sample_indices = np.random.choice(range(len(self.buffer)),
                                          size=batch_size,
                                          replace=False)
        return np.asarray(self.buffer)[sample_indices]

    def add(self, sars_tuples):
        self.buffer += sars_tuples
        if len(self.buffer) > self.buffer_limit:
            self.buffer = self.buffer[-self.buffer_limit:]


class DQNAgent(AgentInterface):
    """
    Algorithm:
        - have parametric value function Q(state) = values_over_actions
        - collect rollouts (s,a,r,s') tuples
        - bootstrap targets: Q(s,a) = r + gamma * max [ Q'(s', *) ]
        -   where Q' is a older Q
        -   MSE training loss
        -   (s,a,r,s') tuples are sampled from a replay memory buffer
        -   look to do batch updates
    """
    def __init__(self, env, config):
        super(DQNAgent, self).__init__(env, config=config)
        approximator_name = config.get(AGENT, "approximator")
        buffer_limit = config.getint(AGENT, "bufferlimit")
        self.batch_size = config.getint(AGENT, "batchsize")
        self.target_update_rate = config.getint(AGENT, "targetupdaterate")
        self.gamma = config.getfloat(AGENT, "gamma")
        self.updates_per_learning_step = config.getint(AGENT, "updatesperstep")

        self.Q_model = load_approximator(approximator_name,
                                         config=config,
                                         inputDim=self.state_space_dim,
                                         outputDim=self.action_space_dim)
        self.target_model = None # until first change over
        self.replay_memory = ReplayMemory(buffer_limit)
        self._copy_target_network()
        self.learning_steps = 0

    def _copy_target_network(self):
        self.target_model = copy(self.Q_model)

    def act(self, ob, reward, done):
        actions_values = self.Q_model.predict(ob)
        if self.train and np.random.random() < self.epsilon:
            return np.random.randint(low=0,
                                     high=self.action_space_dim)
        return np.argmax(actions_values)

    def _get_targets(self):
        batch = self.replay_memory.sample(batch_size=self.batch_size)
        q_targets = []
        states = []
        for sars in batch:
            s, a, r, sprime, done = tuple(sars)  # unpack
            tar_i = self.Q_model.predict(s).flatten()  # prediction from current net
            tar_i[a] = r
            if not done:
                action_values = self.target_model.predict(sprime)   # prediction from target net
                tar_i[a] += self.gamma * np.max(action_values)
            q_targets.append(tar_i)
            states.append(s)
        return np.asarray(states), np.squeeze(np.asarray(q_targets))

    def _update_learning_parameters(self):
        self.learning_steps += 1
        self.Q_model.plot_model(self.model_name)
        if self.learning_steps % self.target_update_rate == 0:
            self._copy_target_network()
            self.epsilon = np.minimum(self.epsilon * self.epsilon_decay_rate,
                                      self.epsilon_limit)

    def _learn(self, sars_tuples):
        self.replay_memory.add(sars_tuples)
        if not self.replay_memory.is_buffer_full():
            return
        logger.info("Learning step: {}".format(self.learning_steps))
        self._update_learning_parameters()
        for i in range(self.updates_per_learning_step):
            x_batch, y_batch = self._get_targets()
            checkpoint_name = os.path.join(MODEL_CHECKPOINTS, self.model_name + HDF5)
            self.Q_model.train(x_batch=x_batch, y_batch=y_batch,
                               batch_size=self.batch_size, model_name=checkpoint_name)

    def end_episode(self, **kwargs):
        self._learn(kwargs["sars_tuples"])
