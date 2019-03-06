import torch
import torch.nn as nn

from src.modules import create_module


class TextModel(nn.Module):
    def __init__(self, config):
        super(TextModel, self).__init__()

        self.config = config

        # TODO: initialize text model

    def forward(self, x, debug=False):
        if debug:
            outputs = []

        # TODO: forward pass

        if debug:
            return x, outputs
        else:
            return x
