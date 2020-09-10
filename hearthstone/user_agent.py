import typing
from typing import List, Optional
from hearthstone.agent import Agent, Action, BuyAction, SummonAction, EndPhaseAction, RerollAction, \
    SellAction, RedeemGoldCoinAction
from hearthstone.agent import TavernUpgradeAction, HeroPowerAction, TripleRewardsAction
from hearthstone.cards import CardLocation
from hearthstone.player import HandIndex, BoardIndex, StoreIndex

if typing.TYPE_CHECKING:
    from hearthstone.cards import Card
    from hearthstone.hero import Hero
    from hearthstone.tavern import Player


class UserAgent(Agent):
    def hero_choice_action(self, player: 'Player') -> 'Hero':
        print(f"player {player.name}, it is your turn to choose a hero.")
        self.print_hero_list(player.hero_options)
        user_text = input("please choose a hero: ")
        while True:
            hero = self.convert_to_hero(user_text, player)
            if hero is not None:
                return hero
            user_text = input("you fucked up, try again: ")

    def convert_to_hero(self, text: str, player: 'Player') -> Optional['Hero']:
        try:
            index = int(text)
        except ValueError:
            return None
        if index in range(len(player.hero_options)):
            return player.hero_options[index]
        return None

    def rearrange_cards(self, player: 'Player'):
        print(f"player {player.name}, it is your combat prephase.")
        self.print_player_card_list("board", player.in_play)
        print("please rearrange your cards by specifying the ordering")
        user_text = input("for example, 1, 0 will swap your 0 and 1 index monsters: ")
        while True:
            arrangement = self.parse_rearrange_input(user_text, player)
            if arrangement:
                return [player.in_play[i] for i in arrangement]
            user_text = input("you fucked up, try again: ")

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

    def buy_phase_action(self, player: 'Player') -> Action:
        print(f"\n\nplayer {player.name} ({player.hero}), it is your buy phase.")
        self.print_player_card_list("store", player.store)
        self.print_player_card_list("board", player.in_play)
        self.print_player_card_list("hand", player.hand)
        print(f"Your current triple rewards are {player.triple_rewards}")
        print(f"you have {player.coins} coins and {player.health} health and your tavern is level {player.tavern_tier}")
        if player.gold_coins >= 1:
            print(f"you have {player.gold_coins} gold coins")
        print("available actions are: ")
        print('purchase: "p 0" purchases the 0th indexed monster from the store')
        print(
            'summon: "s 0 1 2" summons the 0th indexed monster from your hand with battlecry targets index 1 and 2 in board card is placed at the end of the board')
        print(
            'redeem: "r 1" sells the 1 indexed monster from the board ')
        print('reroll store: "R" will reroll the store')
        print(f'upgrade tavern: "u" will upgrade the tavern (current upgrade cost: {player.tavern_upgrade_cost if player.tavern_tier < 6 else 0})')
        print('hero power: "h 0" will activate your hero power with ability target index 0 on the board or in the store')
        print('triple rewards: "t" will use your highest tavern tier triple rewards')
        if player.gold_coins >= 1:
            print('coin tokens: "c" will use a coin token')
        print('end turn: "e f" ends the turn and freezes the shop, "e" ends the turn without freezing the shop')
        user_input = input("input action here: ")
        while True:
            buy_action = self.parse_buy_input(user_input, player)
            if buy_action and buy_action.valid(player):
                return buy_action
            user_input = input("sorry, my dude. Action invalid: ")

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
            if not player.valid_store_index(store_index) or player.coins < player.minion_cost:
                return None
            return BuyAction(StoreIndex(store_index))
        elif split_list[0] == "s":
            if not 2 <= len(split_list) < 5:
                return None
            try:
                targets = [int(target) for target in split_list[1:]]
            except ValueError:
                return None
            if not player.valid_hand_index(targets[0]):
                return None
            for target in targets[1:]:
                if not 0 <= target < len(player.in_play) + 1:
                    return None
            return SummonAction(HandIndex(targets[0]), [BoardIndex(target) for target in targets[1:]])
        elif split_list[0] == "r":
            if not len(split_list) == 2:
                return None
            try:
                sell_index = int(split_list[1])
            except ValueError:
                return None
            if not player.valid_board_index(sell_index):
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
            if not 1 <= len(split_list) < 3:
                return None
            try:
                index = int(split_list[1])
            except IndexError:
                return HeroPowerAction()
            except ValueError:
                return None
            if player.hero.power_target_location == CardLocation.BOARD and index is not None:
                if not player.valid_board_index(index):
                    return None
                return HeroPowerAction(board_target=BoardIndex(index))
            elif player.hero.power_target_location == CardLocation.STORE and index is not None:
                if not player.valid_store_index(index):
                    return None
                return HeroPowerAction(store_target=StoreIndex(index))
        elif split_list[0] == "t":
            return TripleRewardsAction()
        elif split_list[0] == "c":
            return RedeemGoldCoinAction()
        else:
            return None

    def discover_choice_action(self, player: 'Player') -> 'Card':
        print(f"player {player.name}, you must choose a card to discover.")
        self.print_player_card_list("discovery choices", player.discovered_cards)
        user_input = input("input card number to discover here: ")
        while True:
            discover_card = self.parse_discover_input(user_input, player)
            if discover_card:
                return discover_card
            user_input = input("oops, try again: ")

    @staticmethod
    def parse_discover_input(user_input: str, player: 'Player') -> Optional['Card']:
        try:
            card_index = int(user_input)
            return player.discovered_cards[card_index]
        except ValueError:
            return None

    @staticmethod
    def print_player_card_list(card_location: str, card_list: List['Card']):
        print(f"your current {card_location}: ")
        for index, card in enumerate(card_list):
            print(index, "  ", card)

    @staticmethod
    def print_hero_list(hero_list: List['Hero']):
        for index, hero in enumerate(hero_list):
            print(index, "  ", hero)
