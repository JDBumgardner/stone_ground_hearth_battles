import unittest

from hearthstone.tavern import Tavern
from hearthstone.training.pytorch.hearthstone_state_encoder import encode_player, encode_valid_actions


class PytorchTests(unittest.TestCase):
    def test_encoding(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("brian")
        tavern.buying_step()
        player_1_encoding = encode_player(player_1)
        print(player_1_encoding)

    def test_valid_actions(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("brian")
        tavern.buying_step()
        player_1_valid_actions = encode_valid_actions(player_1)
        print(player_1_valid_actions)

if __name__ == '__main__':
    unittest.main()
