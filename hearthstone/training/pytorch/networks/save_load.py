from typing import Dict

import torch
from torch import nn

from hearthstone.training.pytorch.hearthstone_state_encoder import DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING
from hearthstone.training.pytorch.networks.feedforward_net import HearthstoneFFNet
from hearthstone.training.pytorch.networks.transformer_net import HearthstoneTransformerNet


def create_net(hparams: Dict) -> nn.Module:
    if hparams["nn.architecture"] == "feedforward":
        return HearthstoneFFNet(DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING,
                                hparams["nn.hidden_layers"],
                                hparams.get("nn.hidden_size") or 0,
                                hparams.get("nn.shared") or False,
                               hparams.get("nn.activation") or "")
    elif hparams["nn.architecture"] == "transformer":
        return HearthstoneTransformerNet(DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING,
                                         hparams["nn.hidden_layers"],
                                         hparams.get("nn.hidden_size") or 0,
                                         hparams.get("nn.shared") or False,
                                         hparams.get("nn.activation") or "",
                                         hparams.get("nn.encoding.redundant"))


def load_from_saved(path, hparams) -> nn.Module:
    net = create_net(hparams)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net