from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from utils.constants import AGENT
import numpy as np


class FeedForwardNet(object):
    def __init__(self, config, inputDim, outputDim):
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.nlayers = config.getint(AGENT, "nlayers")
        hidden_activations = config.get(AGENT, "hiddenactivations")
        output_activation = config.get(AGENT, "outputactivation")
        optimiser = config.get(AGENT, "optimiser")
        loss = config.get(AGENT, "loss")
        # create the model:
        self._set_layer_sizes()
        self.model = Sequential()
        for layer_i in range(self.nlayers):
            if layer_i == self.nlayers - 1:
                # Output layer
                self.model.add(Dense(output_dim=self.layer_sizes[layer_i + 1],
                                     input_dim=self.layer_sizes[layer_i],
                                     activation=Activation(output_activation)))
            else:
                # Hidden layer
                self.model.add(Dense(output_dim=self.layer_sizes[layer_i + 1],
                                     input_dim=self.layer_sizes[layer_i],
                                     activation=Activation(hidden_activations)))

        self.model.compile(optimizer=optimiser,
                           loss=loss)

    def _set_layer_sizes(self):
        """
        sets field layer_sizes to list [stateSpaceDim | h1 | h2 | ... | actionSpaceDim]
        """
        self.layer_sizes = [self.inputDim]
        h_1 = self.inputDim
        for i in range(self.nlayers - 1):
            h = np.floor(0.5 * h_1)
            h = h if h >= self.outputDim else self.outputDim
            self.layer_sizes.append(h)
            h_1 = h
        self.layer_sizes.append(self.outputDim)

    def predict(self, x_batch, batch_size=1):
        return self.model.predict(x=x_batch.reshape(batch_size, x_batch.size),
                                  batch_size=batch_size)

    def train(self, x_batch, y_batch, batch_size):
        self.model.fit(x=x_batch,
                       y=y_batch,
                       batch_size=batch_size)
