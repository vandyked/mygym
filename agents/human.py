from agent_interface import AgentInterface
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import getch


class HumanAgent(AgentInterface):
    """
    Uses getch() from https://pypi.python.org/pypi/getch
    """
    def __init__(self, env, config):
        env.mode = "human"  # slower rendering for human play
        # need to setup method to fix desired key bindings for each game
        self.keybindings = dict(config.items('keybindings'))
        for key in self.keybindings.keys():
            self.keybindings[key] = int(self.keybindings[key])
        logger.info("Keys for game are: {}".format(self.keybindings))

    def act(self, ob, reward, done):
        print ":",
        key = getch.getche()  # getch.getch() doesn't output to screen
        action = self.keybindings.get(key)
        if action is None:
            logger.warning("Invalid key. Valid keys: {}".format(self.keybindings))
            action = self.act(ob, reward, done)
        return action

