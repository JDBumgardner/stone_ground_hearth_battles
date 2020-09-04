from typing import Optional, List

from hearthstone.agent import SummonAction, SellAction, EndPhaseAction, RerollAction, TavernUpgradeAction, \
    HeroPowerAction, TripleRewardsAction, RedeemGoldCoinAction, BuyAction, Action, Agent
from hearthstone.player import HandIndex, BoardIndex, StoreIndex


class TextAgentTransport:
    async def receive_line(self) -> str:
        pass

    async def send(self, text: str):
        pass


class TextAgent(Agent):
    def __init__(self, transport: TextAgentTransport):
        self.transport = transport

    async def hero_choice_action(self, player: 'Player') -> 'Hero':
        await self.transport.send(f"player {player.name}, it is your turn to choose a hero.\n")
        await self.print_hero_list(player.hero_options)
        await self.transport.send("please choose a hero: ")
        user_text = await self.transport.receive_line()
        while True:
            hero = self.convert_to_hero(user_text, player)
            if hero is not None:
                return hero
            await self.transport.send("you fucked up, try again: ")
            user_text = await self.transport.receive_line()

    def convert_to_hero(self, text: str, player: 'Player') -> Optional['Hero']:
        try:
            index = int(text)
        except ValueError:
            return None
        if index in range(len(player.hero_options)):
            return player.hero_options[index]
        return None

    async def rearrange_cards(self, player: 'Player'):
        await self.transport.send(f"player {player.name}, it is your combat prephase.\n")
        await self.print_player_card_list("board", player.in_play)
        await self.transport.send("please rearrange your cards by specifying the ordering\n")
        await self.transport.send("for example, 1, 0 will swap your 0 and 1 index monsters: ")
        user_text = await self.transport.receive_line()
        while True:
            arrangement = self.parse_rearrange_input(user_text, player)
            if arrangement:
                return [player.in_play[i] for i in arrangement]
            await self.transport.send("you fucked up, try again: ")
            user_text = await self.transport.receive_line()

    @staticmethod
    def parse_rearrange_input(user_text: str, player: 'Player') -> Optional[List[int]]:
        split_list = user_text.split(',')
        if split_list == ["a"]:
           return list(range(len(player.in_play)))
        try:
            check_list = [int(i) for i in split_list]
        except ValueError:
            return None
        check_items = range(len(player.in_play))
        if set(check_list) != set(check_items):
            return None
        return check_list

    async def buy_phase_action(self, player: 'Player') -> Action:
        await self.transport.send(f"player {player.name} ({player.hero}), it is your buy phase.\n")
        await self.print_player_card_list("store", player.store)
        await self.print_player_card_list("board", player.in_play)
        await self.print_player_card_list("hand", player.hand)
        await self.transport.send(f"Your current triple rewards are {player.triple_rewards}\n")
        await self.transport.send(f"you have {player.coins} coins and {player.health} health and your tavern is level {player.tavern_tier}\n")
        if player.gold_coins >= 1:
            print(f"you have {player.gold_coins} gold coins\n")
        await self.transport.send("available actions are: \n")
        await self.transport.send('purchase: "p 0" purchases the 0th indexed monster from the store\n')
        await self.transport.send(
            'summon: "s 0 1 2" summons the 0th indexed monster from your hand with ability targets index 1 and 2 in board card is placed at the end of the board\n')
        await self.transport.send(
            'redeem: "r h 1" sells the 1 indexed monster from hand "r b 2" sells the 2 indexed monster from the board \n')
        await self.transport.send('reroll store: "R" will reroll the store\n')
        await self.transport.send(f'upgrade tavern: "u" will upgrade the tavern (current upgrade cost: {player.tavern_upgrade_cost if player.tavern_tier < 6 else 0})\n')
        await self.transport.send('hero power: "h" will activate your hero power\n')
        await self.transport.send('triple rewards: "t" will use your highest tavern tier triple rewards\n')
        if player.gold_coins >= 1:
            await self.transport.send('coin tokens: "c" will use a coin token\n')
        await self.transport.send('end turn: "e f" ends the turn and freezes the shop, "e" ends the turn without freezing the shop\n')
        await self.transport.send("input action here: ")
        user_input = await self.transport.receive_line()
        while True:
            buy_action = self.parse_buy_input(user_input, player)
            if buy_action and buy_action.valid(player):
                return buy_action
            await self.transport.send("sorry, my dude. Action invalid: ")
            user_input = await self.transport.receive_line()

    @staticmethod
    def parse_buy_input(user_input: str, player: 'Player') -> Optional[Action]:
        split_list = user_input.split(" ")
        if split_list[0] == "p":
            if not len(split_list) == 2:
                return None
            try:
                store_index = int(split_list[1])
            except ValueError:
                return None
            if not 0 <= store_index < len(player.store) or player.coins < player.minion_cost:
                return None
            return BuyAction(StoreIndex(store_index))
        elif split_list[0] == "s":
            if not 1 < len(split_list) <= 4:
                return None
            try:
                targets = [int(target) for target in split_list[1:]]
            except ValueError:
                return None
            if not 0 <= targets[0] < len(player.hand):
                return None
            for target in targets[1:]:
                if not 0 <= target < len(player.in_play) + 1:
                    return None
            in_play = player.in_play + [player.hand[targets[0]]]
            return SummonAction(HandIndex(targets[0]), [BoardIndex(target) for target in targets[1:]])
        elif split_list[0] == "r":
            if not len(split_list) == 2:
                return None
            try:
                sell_index = int(split_list[1])
            except ValueError:
                return None
            if not 0 <= sell_index < len(player.in_play):
                return None
            return SellAction(BoardIndex(sell_index))
        elif split_list == ["e"]:
            return EndPhaseAction(False)
        elif split_list == ["e", "f"]:
            return EndPhaseAction(True)
        elif split_list[0] == "R":
            return RerollAction()
        elif split_list[0] == "u":
            return TavernUpgradeAction()
        elif split_list[0] == "h":
            return HeroPowerAction()
        elif split_list[0] == "t":
            return TripleRewardsAction()
        elif split_list[0] == "c":
            return RedeemGoldCoinAction()
        else:
            return None

    async def discover_choice_action(self, player: 'Player') -> 'Card':
        await self.transport.send(f"player {player.name}, you must choose a card to discover.")
        await self.print_player_card_list("discovery choices", player.discovered_cards)
        await self.transport.send("input card number to discover here: ")
        user_input = self.transport.receive_line()
        while True:
            discover_card = self.parse_discover_input(user_input, player)
            if discover_card:
                return discover_card
            await self.transport.send("oops, try again: ")
            user_input = self.transport.receive_line()

    @staticmethod
    async def parse_discover_input(user_input: str, player: 'Player') -> Optional['Card']:
        try:
            card_index = int(user_input)
            return player.discovered_cards[card_index]
        except ValueError:
            return None

    async def print_player_card_list(self, card_location: str, card_list: List['Card']):
        await self.transport.send(f"your current {card_location}: \n")
        for index, card in enumerate(card_list):
            await self.transport.send(f"{index}  {card}\n")

    async def print_hero_list(self, hero_list: List['Hero']):
        for index, hero in enumerate(hero_list):
            await self.transport.send(f"{index} {hero}\n")