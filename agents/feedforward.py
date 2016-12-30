from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from agent_interface import AgentInterface
import numpy as np

class KerasFeedForwardAgent(AgentInterface):
    def __init__(self, env, config):
        super(KerasFeedForwardAgent, self).__init__(env)
        self.nlayers = config.getint("agent", "nlayers")

        # create the model:
        self._set_layer_sizes()
        self.model = Sequential()
        for layer_i in range(len(self.layer_sizes)-1):
            if layer_i != len(self.layer_sizes)-1:
                self.model.add(Dense(output_dim=self.layer_sizes[layer_i + 1],
                                     input_dim=self.layer_sizes[layer_i],
                                     activation=Activation('relu')))
            else:
                self.model.add(Dense(output_dim=self.layer_sizes[layer_i + 1],
                                     input_dim=self.layer_sizes[layer_i],
                                     activation=Activation('softmax')))

        # UP TO HERE
        need to look at training by either:
            -   DQN with bootstrapping of value function with a replay mem
            -   policy gradient

        self.model.compile(optimizer='adam',
                           loss=
                           )

    def _set_layer_sizes(self):
        """
        sets field layer_sizes to list [stateSpaceDim | h1 | h2 | ... | actionSpaceDim]
        """
        self.layer_sizes = [self.state_space_dim]
        for i in range(self.nlayers-1):
            h = np.floor(0.5 * self.state_space_dim)
            h = h if h >= self.action_space_dim else self.action_space_dim
            self.layer_sizes.append(h)
        self.layer_sizes.append(self.action_space_dim)