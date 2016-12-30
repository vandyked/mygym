class AgentInterface(object):
    def __init__(self, env, config=None):
        self.action_space = env.action_space
        try:
            if hasattr(env, 'state'):
                self.state_space_shape = env.state.shape
            elif hasattr(env, 'observation_space'):
                self.state_space_shape = env.observation_space.shape
        except:
            exit("Failed to get state dimension from environment")


    def act(self, ob, reward, done):
        """ RandomAgent for interface
        :param ob:
        :param reward:
        :param done:
        :return:
        """
        return self.action_space.sample()

    def cleanup(self, **kwargs):
        pass

    def start_episode(self):
        pass

    def end_episode(self, **kwargs):
        pass