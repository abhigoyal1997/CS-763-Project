import torch
import torch.nn as nn

from src.modules import create_module


class ImageModel(nn.Module):
    def __init__(self, config, args):
        super(ImageModel, self).__init__()
        self.config = config

        self.in_shape = args['in_shape']

        x = torch.rand(1, *self.in_shape)
        self.layers = nn.ModuleList()
        i = 1
        while i < len(config):
            self.layers.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x = self.layers[-1](x)
            i += 1

    def forward(self, x, z=None, debug=False):
        if debug:
            outputs = []

        for layer in self.layers:
            x = layer(x)
            if debug:
                outputs.append(x)

        if debug:
            return x, outputs
        else:
            return x
