from hearthstone.simulator.agent import StandardAction
from hearthstone.simulator.replay.replay import Replay
from hearthstone.training.pytorch.replay import GAEReplayInfo


class GAEAnnotator:
    def __init__(self, player: str, gamma: float, lam: float):
        self.player = player
        self.gamma = gamma
        self.lam = lam

    def annotate(self, replay: Replay):
        reward = (len(replay.players) - 1) / 2.0 - replay.agent_annotations[self.player]['ranking']
        retrn = reward
        gae_return = reward
        next_value = reward
        # We iterate backwards over the actions taken by this player only.  For now, we are not learning from nonstandard actions.
        reversed_player_steps = reversed([game_step for game_step in replay.steps if
                                          game_step.player == self.player and isinstance(game_step.action,
                                                                                         StandardAction)])
        for i, game_step in enumerate(reversed_player_steps):
            is_terminal = i == 0
            game_step.agent_annotation.gae_info = GAEReplayInfo(
                is_terminal=is_terminal,
                reward=reward if is_terminal else 0,
                gae_return=gae_return,
                retrn=retrn,
            )
            gae_return = next_value + (gae_return - next_value) * self.gamma * self.lam
            next_value = self.gamma * game_step.agent_annotation.value
            retrn *= self.gamma
