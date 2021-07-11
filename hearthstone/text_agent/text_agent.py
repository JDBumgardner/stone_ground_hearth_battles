from typing import Optional, List, Union

from hearthstone.simulator.agent.actions import HeroChoiceAction, RearrangeCardsAction, StandardAction, BuyAction, \
    SummonAction, SellAction, EndPhaseAction, FreezeDecision, RerollAction, TavernUpgradeAction, HeroPowerAction, \
    DiscoverChoiceAction, PlaySpellAction
from hearthstone.simulator.agent.agent import Agent
from hearthstone.simulator.core.cards import CardLocation, MonsterCard
from hearthstone.simulator.core.hero import Hero
from hearthstone.simulator.core.player import HandIndex, BoardIndex, StoreIndex, Player, DiscoverIndex, HeroChoiceIndex, \
    SpellIndex
from hearthstone.simulator.core.spell import Spell


class TextAgentProtocol:
    async def receive_line(self) -> str:
        pass

    async def send(self, text: str):
        pass


class TextAgent(Agent):
    def __init__(self, connection: TextAgentProtocol):
        self.connection = connection

    async def hero_choice_action(self, player: 'Player') -> 'HeroChoiceAction':
        await self.connection.send(f"player {player.name}, it is your turn to choose a hero.\n")
        await self.connection.send(
            f"available monster types: {[monster_type.name for monster_type in player.tavern.available_types]}\n")
        await self.print_hero_list(player.hero_options)
        await self.connection.send("please choose a hero: ")
        user_text = await self.connection.receive_line()
        while True:
            hero_choice_action = self.convert_to_hero(user_text, player)
            if hero_choice_action is not None:
                return hero_choice_action
            await self.connection.send("you fucked up, try again: ")
            user_text = await self.connection.receive_line()

    def convert_to_hero(self, text: str, player: 'Player') -> Optional['HeroChoiceAction']:
        try:
            index = int(text)
        except ValueError:
            return None
        if index in range(len(player.hero_options)):
            return HeroChoiceAction(HeroChoiceIndex(index))
        return None

    async def rearrange_cards(self, player: 'Player') -> 'RearrangeCardsAction':
        await self.connection.send(f"player {player.name}, it is your combat prephase.\n")
        await self.print_player_card_list("board", player.in_play)
        await self.connection.send("please rearrange your cards by specifying the ordering\n")
        await self.connection.send("for example, 1, 0 will swap your 0 and 1 index monsters\n")
        await self.connection.send("a will accept your current board order: ")
        user_text = await self.connection.receive_line()
        while True:
            arrangement = self.parse_rearrange_input(user_text, player)
            if arrangement:
                return RearrangeCardsAction(arrangement)
            await self.connection.send("you fucked up, try again: ")
            user_text = await self.connection.receive_line()

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

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        await self.connection.send(f"\n\nplayer {player.name} ({player.hero}), it is your buy phase.\n")
        await self.connection.send(
            f"available monster types: {[monster_type.name for monster_type in player.tavern.available_types]}\n")
        await self.connection.send(f"\nCurrent standings:\n")
        for i, contenstant in enumerate(
                sorted(player.tavern.players.values(), key=lambda plyr: plyr.health, reverse=True)):
            build = contenstant.current_build()
            if build == (None, None):
                build = "Mixed Minions"
            else:
                build = str(build[0]) + 'S ' + str(build[1])
            await self.connection.send(
                f"{i + 1}: {str(contenstant):<53} [{str(max(contenstant.health, 0)) + ' health,':<12} tier {contenstant.tavern_tier},\t {build}]\n")
        await self.connection.send(f"\nYour next opponent: {player.next_opponent().hero}\n\n")
        await self.print_player_card_list("store", player.store)
        await self.print_player_card_list("board", player.in_play)
        await self.print_player_card_list("hand", player.hand)
        await self.print_player_card_list("spells", player.spells)
        await self.connection.send(
            f"you have {player.coins} coins and {player.health} health and your tavern is level {player.tavern_tier}\n")
        await self.connection.send("available actions are: \n")
        await self.connection.send('purchase: "p 0" purchases the 0th indexed monster from the store\n')
        await self.connection.send(
            'summon: "s 0 [1] [2]" summons the 0th indexed monster from your hand with battlecry targets index 1 and 2 in board card is placed at the end of the board\n')
        await self.connection.send('redeem: "r 1" sells the 1 indexed monster from the board\n')
        await self.connection.send('reroll store: "R" will reroll the store\n')
        await self.connection.send(
            f'upgrade tavern: "u" will upgrade the tavern (current upgrade cost: {player.tavern_upgrade_cost if player.tavern_tier < 6 else 0})\n')
        await self.connection.send(
            f'hero power: "h [b/s] [0]" will activate your hero power with ability target index 0 on the board/store (current cost: {player.hero.power_cost})\n')
        await self.connection.send(
            f'hero power: "c 0 [b/s] [0]" plays the 0th indexed spell with ability target index 0 on the board/store\n')
        if player.hero.hero_info(player) is not None:
            await self.connection.send(f'hero info: {player.hero.hero_info(player)}\n')
        await self.connection.send(
            'end turn: "e f" ends the turn and freezes the shop, "e u" ends the turn and unfreezes the shop, "e" ends the turn with no changes\n')
        await self.connection.send("input action here: ")
        user_input = await self.connection.receive_line()
        while True:
            buy_action = self.parse_buy_input(user_input, player)
            if buy_action and buy_action.valid(player):
                return buy_action
            await self.connection.send("sorry, my dude. Action invalid: ")
            user_input = await self.connection.receive_line()

    @staticmethod
    def parse_buy_input(user_input: str, player: 'Player') -> Optional[StandardAction]:
        split_list = user_input.split(" ")
        if split_list[0] == "p":
            if not len(split_list) == 2:
                return None
            try:
                store_index = int(split_list[1])
            except ValueError:
                return None
            if not player.valid_store_index(StoreIndex(store_index)) or player.coins < player.minion_cost:
                return None
            return BuyAction(StoreIndex(store_index))
        elif split_list[0] == "s":
            if not 2 <= len(split_list) < 5:
                return None
            try:
                targets = [int(target) for target in split_list[1:]]
            except ValueError:
                return None
            if not player.valid_hand_index(HandIndex(targets[0])):
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
            if not player.valid_board_index(BoardIndex(sell_index)):
                return None
            return SellAction(BoardIndex(sell_index))
        elif split_list == ["e"]:
            return EndPhaseAction(FreezeDecision.NO_FREEZE)
        elif split_list == ["e", "f"]:
            return EndPhaseAction(FreezeDecision.FREEZE)
        elif split_list == ["e", "u"]:
            return EndPhaseAction(FreezeDecision.UNFREEZE)
        elif split_list[0] == "R":
            return RerollAction()
        elif split_list[0] == "u":
            return TavernUpgradeAction()
        elif split_list[0] == "h":
            if not 1 <= len(split_list) < 4:
                return None
            try:
                location = split_list[1]
            except IndexError:
                return HeroPowerAction()
            try:
                index = int(split_list[2])
            except (ValueError, IndexError):
                return None
            if player.hero.power_target_location is not None:
                if CardLocation.BOARD in player.hero.power_target_location and location == "b":
                    if not player.valid_board_index(BoardIndex(index)):
                        return None
                    return HeroPowerAction(board_target=BoardIndex(index))
                elif CardLocation.STORE in player.hero.power_target_location and location == "s":
                    if not player.valid_store_index(StoreIndex(index)):
                        return None
                    return HeroPowerAction(store_target=StoreIndex(index))
        elif split_list[0] == "c":
            if not 2 <= len(split_list) < 5:
                return None
            try:
                index = int(split_list[1])
            except (ValueError, IndexError):
                return None
            try:
                location = split_list[2]
            except IndexError:
                return PlaySpellAction(SpellIndex(index))
            try:
                target_index = int(split_list[3])
            except (ValueError, IndexError):
                return None
            if location == "b":
                if not player.valid_board_index(BoardIndex(target_index)):
                    return None
                return PlaySpellAction(SpellIndex(index), board_target=BoardIndex(target_index))
            elif location == "s":
                if not player.valid_store_index(StoreIndex(target_index)):
                    return None
                return PlaySpellAction(SpellIndex(index), store_target=StoreIndex(target_index))
        else:
            return None

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        await self.connection.send(f"player {player.name}, you must choose a card to discover.\n")
        await self.print_player_card_list("discovery choices", player.discover_queue[0].items)
        await self.connection.send("input card number to discover here: ")
        user_input = await self.connection.receive_line()
        while True:
            discover_card = self.parse_discover_input(user_input, player)
            if discover_card:
                return discover_card
            await self.connection.send("oops, try again: ")
            user_input = await self.connection.receive_line()

    @staticmethod
    def parse_discover_input(user_input: str, player: 'Player') -> Optional['DiscoverChoiceAction']:
        if not user_input.isnumeric():
            return None
        card_index = int(user_input)
        if card_index in range(len(player.discover_queue[0].items)):
            return DiscoverChoiceAction(DiscoverIndex(card_index))
        else:
            return None

    async def print_player_card_list(self, card_location: str, card_list: List[Union['MonsterCard', 'Spell']]):
        await self.connection.send(f"your current {card_location}: \n")
        for index, card in enumerate(card_list):
            await self.connection.send(f"{index}  {card}\n")

    async def print_hero_list(self, hero_list: List['Hero']):
        for index, hero in enumerate(hero_list):
            await self.connection.send(f"{index} {hero}\n")

    async def game_over(self, player: 'Player', ranking: int):
        await self.connection.send(
            f'\n\n**************you have been killed you were ranked #{ranking}*******************')
