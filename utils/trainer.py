from utils.constants import TRAINER
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, config, agent, env):
        self.agent = agent
        self.env = env
        self.iterations = config.getint(TRAINER, "iterations")
        self.max_steps = config.getint(TRAINER, "maxsteps")
        self.render = config.getboolean(TRAINER, "render")
        self.plot_frequency = config.getint(TRAINER, "plotevery")
        try:
            self.output_dir = config.get(TRAINER, "outdir")
        except:
            self.output_dir = os.path.join(['outdir', self.env.__name__])
        self.episode_reward_history = []

    def setup(self):
        if self.render:
            logger.info("Start recording to: {}".format(self.output_dir))
            self.env.monitor.start(self.output_dir, force=True)

    def cleanup(self):
        self.agent.cleanup()
        if self.render:
            logger.info("End recording. Successfully completed rollout")
            self.env.monitor.close()

    def train(self):
        for loopNum in range(self.iterations):
            self.agent.start_episode()
            training_info = self.do_rollout(self.max_steps, self.render)
            logger.info("Episode: {}\tsteps: {}\ttotal_reward: {}".format(
                loopNum,
                len(training_info["sars_tuples"]),
                training_info["total_reward"]))
            self.agent.end_episode(**training_info)
            if self.plot_frequency and loopNum and loopNum % self.plot_frequency == 0:
                self.plot_reward_history()

    def do_rollout(self, max_num_steps=10, render=False):
        """Generic function (modified from gym examples) that will work for most agent:
        :param max_num_steps: limit on number of steps in episode
        :param render: boolean
        :return: {total_reward, sars_tuples}
        """
        total_rew = 0
        # Initial observations from environment
        ob = self.env.reset()
        reward = 0
        done = False
        sars_tuples = []

        for t in range(max_num_steps):
            current_tuple = []
            a = self.agent.act(ob, reward, done)
            current_tuple += [ob, a]
            (ob, reward, done, _info) = self.env.step(a)
            current_tuple += [reward, ob, done]
            sars_tuples.append(tuple(current_tuple))
            total_rew += reward
            if render and t%3==0: self.env.render()
            if done: break

        self.episode_reward_history.append(total_rew)
        return {"total_reward": total_rew, "sars_tuples": sars_tuples}

    def plot_reward_history(self):
        from utils.ploting import plot_data
        plot_data(x_data=[range(len(self.episode_reward_history))],
                  y_data=[self.episode_reward_history])



