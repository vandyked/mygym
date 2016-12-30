from utils.rollout import setup_env
from utils.rollout import teardown_env
from utils.rollout import do_rollout
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
        setup_env(self.env, self.outdir)

    def cleanup(self):
        self.agent.cleanup()
        teardown_env(self.env)

    def train(self):
        for loopNum in range(self.iterations):
            self.agent.start_episode()
            training_info = do_rollout(self.env, self.agent, self.max_steps, self.render)
            logger.debug("Episode: {} lasted: {}".format(loopNum, len(training_info["sars_tuples"])))
            self.agent.end_episode(**training_info)
