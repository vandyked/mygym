from keras.optimizers import Adam
from utils.constants import AGENT


def load_optimiser(config):
    optimiser_name = config.get(AGENT, "optimiser")
    if config.has_option(AGENT, "learningrate"):
        lr = config.getfloat(AGENT, "learningrate")

    if optimiser_name == "adam":
        """ except lr, others are default params
        """
        return Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:
        exit("unknown optimiser: {}".format(optimiser_name))
