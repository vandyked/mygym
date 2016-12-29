import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from utils.rollout import setup_env
from utils.rollout import teardown_env
from utils.rollout import do_rollout

TRAINER = "trainer"

class Trainer(object):
    def __init__(self, config, agent, env):
        self.agent = agent
        self.env = env
        self.iterations = config.getint(TRAINER, "iterations")
        self.max_steps = config.getint(TRAINER, "maxsteps")
        self.render = config.getboolean(TRAINER, "render")
        self.outdir = config.get(TRAINER, "outdir")

    def setup(self):
        self.agent.setup()
        setup_env(self.env, self.outdir)

    def cleanup(self):
        self.agent.cleanup()
        teardown_env(self.env)

    def train(self):
        self.agent.setup()
        for loopi in range(self.iterations):
            total_rew, sars_tuples = do_rollout(self.env, self.agent, self.max_steps, self.render)
            logger.debug("Episode: {} lasted: {}".format(loopi, len(sars_tuples)))
            self.agent.training_step(total_rew, sars_tuples)
