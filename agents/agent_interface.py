class AgentInterface(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self._setup()
    def _setup(self, **kwargs):
        pass
    def act(self, ob, reward, done):
        """ RandomAgent for interface
        :param ob:
        :param reward:
        :param done:
        :return:
        """
        return self.action_space.sample()
    def setup(self, **kwargs):
        pass
    def cleanup(self, **kwargs):
        pass
    def training_step(self, **kwargs):
        pass