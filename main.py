import gym
import ConfigParser
import argparse
import logging

from utils.load_agents import load_agent
from utils.trainer import Trainer

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', '-c', default=None)
    parser.add_argument('--help', '-h', default=None)
    args = parser.parse_args()

    if args.help is not None or args.config is None:
        exit("Usage: python main.py -c configFile [-h help]")

    config = ConfigParser.ConfigParser()
    config.read(args.config)
    env_name = config.get("env", "id")
    agent_name = config.get("agent", "id")

    env = gym.make(env_name)
    agent = load_agent(agent_name, env, config)
    trainer = Trainer(config, agent, env)
    trainer.setup()
    trainer.train()
    trainer.cleanup()




