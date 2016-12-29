import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from agents.agent_interface import AgentInterface
from agents.cem import CEMAgent

def load_agent(agent_name, action_space):
    if agent_name == AgentInterface.__name__:
        logger.debug("Loading AgentInterface/RandomAgent")
        return AgentInterface(action_space)
    elif agent_name == CEMAgent.__name__:
        logger.debug("Loading black box Cross-Entropy Agent")
        return CEMAgent(action_space)
    else:
        logger.error("No agent named: {}".format(agent_name))


def setup_env(env, output_dir=None):
    if output_dir is None:
        output_dir = 'output/' + env.__name__
    logger.info("Start recording to: {}".format(output_dir))
    env.monitor.start(output_dir, force=True)


def teardown_env(env):
    logger.info("End recording. Successfully completed rollout")
    env.monitor.close()


def do_rollout(env, agent, num_steps, render=False):
    """Generic function (modified from gym examples) that will work for most agent:
    :param agent:
    :param num_steps:
    :param render:
    :return:
    """
    total_rew = 0
    # Initial observations from environment
    ob = env.reset()
    reward = 0
    done = False
    sars_tuples = []

    for t in range(num_steps):
        current_tuple = []
        a = agent.act(ob, reward, done)
        current_tuple += [ob, a]
        (ob, reward, done, _info) = env.step(a)
        current_tuple += [reward, ob, done]
        sars_tuples.append(tuple(current_tuple))
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, sars_tuples
