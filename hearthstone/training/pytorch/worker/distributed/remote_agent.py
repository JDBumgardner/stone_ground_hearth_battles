from torch.distributed.rpc import RRef

from hearthstone.simulator.agent import AnnotatingAgent, Annotation, DiscoverChoiceAction, StandardAction, \
    RearrangeCardsAction, HeroChoiceAction


class RemoteAgent(AnnotatingAgent):
    def __init__(self, remote_agent: RRef):
        self.remote_agent = remote_agent

    async def hero_choice_action(self, player: 'Player') -> HeroChoiceAction:
        return self.remote_agent.rpc_sync().hero_choice_action(player)

    async def annotated_rearrange_cards(self, player: 'Player') -> (RearrangeCardsAction, Annotation):
        return self.remote_agent.rpc_sync().annotated_rearrange_cards(player)

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, Annotation):
        return self.remote_agent.rpc_sync().annotated_buy_phase_action(player)

    async def annotated_discover_choice_action(self, player: 'Player') -> (DiscoverChoiceAction, Annotation):
        return self.remote_agent.rpc_sync().annotated_discover_choice_action(player)

    async def annotated_hero_discover_action(self, player: 'Player') -> ('HeroDiscoverAction', Annotation):
        return self.remote_agent.rpc_sync().annotated_hero_discover_action(player)

    async def game_over(self, player: 'Player', ranking: int) -> Annotation:
        return self.remote_agent.rpc_sync().game_over(player, ranking)
