from approximators.feedforward import FeedForwardNet

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_approximator(approximator_name, config, inputDim, outputDim):
    if approximator_name == FeedForwardNet.__name__:
        logger.debug("Loading feed forward approximator")
        return FeedForwardNet(config, inputDim, outputDim)
    else:
        logger.error("No approximator named: {}".format(approximator_name))
        exit()