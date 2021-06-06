from gym.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from hearthstone.simulator.host.async_host import AsyncHost


def env():
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {'render.modes': ['human'], "name": "sghb"}

    def __init__(self, n_players=8):
        self.possible_agents = ["player_" + str(r) for r in range(n_players)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        self.observation_spaces = {agent: Discrete(4) for agent in self.possible_agents}

    def render(self, mode="human"):
        print("Rendered, noob")

    def close(self):
        pass

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.host = AsyncHost()
        self.agent_selection = self._agent_selector.next()
