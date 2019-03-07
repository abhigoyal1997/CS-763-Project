import torch
import torch.nn as nn

from src.modules import create_module


class TextModel(nn.Module):
    def __init__(self, config, args):
        super(TextModel, self).__init__()

        self.config = config

        self.embed = create_module(config[1], args['vocab_size'])
        self.vocab_size = self.embed.num_embeddings
        self.rcell = create_module(config[2], config[1][1])

        self.layers = nn.ModuleList()
        i = 3
        while i < len(config):
            self.layers.append(create_module(config[i], config[i-1][1]))
            i += 1

        self.is_cuda = False

    @property
    def out_shape(self):
        if self.is_cuda:
            return self.forward(torch.Tensor([[1]]).long().cuda(), torch.Tensor([1]).long().cuda()).shape[-1]
        else:
            return self.forward(torch.Tensor([[1]]).long(), torch.Tensor([1]).long()).shape[-1]

    def cuda(self, device=None):
        self.is_cuda = True
        return super(TextModel, self).cuda(device)

    def forward(self, x, lengths, debug=False):
        if debug:
            outputs = []

        x = self.embed(x)
        if debug:
            outputs.append(x)

        x = nn.utils.rnn.pack_padded_sequence(x,lengths)
        if isinstance(self.rcell, nn.LSTM):
            _, (_,c) = self.rcell(x)
        elif isinstance(self.rcell, nn.RNN):
            _,c = self.rcell(x)
        x = c[-1]

        if debug:
            outputs.append(x)

        for layer in self.layers:
            x = layer(x)
            if debug:
                outputs.append(x)

        if debug:
            return x, outputs
        else:
            return x
