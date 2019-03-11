import torch
import torch.nn as nn
import numpy as np


class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)


MODULES = {
    'nn': {
        'conv': nn.Conv2d,
        'linear': nn.Linear,
        'norm1d': nn.BatchNorm1d,
        'norm2d': nn.BatchNorm2d,
        'lstm': nn.LSTM,
        'rnn': nn.RNN,
    },
    'f': {
        'max2d': nn.MaxPool2d,
        'relu': nn.ReLU,
        'drop1d': nn.Dropout,
        'drop2d': nn.Dropout2d,
        'flat': Flat,
        'pad2d': nn.ZeroPad2d
    }
}


def pretrained_embedding_layer(args):
    emb = np.load(args[0])['embeddings']
    embeddings = torch.Tensor(emb)
    num_embeddings, embedding_dim = embeddings.size()
    layer = nn.Embedding(num_embeddings, embedding_dim)
    layer.load_state_dict({'weight': embeddings})
    layer.weight.requires_grad = False
    return layer


def create_module(config, in_features=None):
    if config[0] == 'embed':
        if config[1] == 'p':
            return pretrained_embedding_layer(config[2:])
        else:
            return nn.Embedding(in_features, *config[1:])

    try:
        if config[0] in MODULES['nn']:
            module = MODULES['nn'][config[0]](in_features, *config[1:])
        else:
            module = MODULES['f'][config[0]](*config[1:])
    except KeyError:
        print('Module {} not found!'.format(config[0]))
        raise KeyError
    except Exception:
        print('Error while creating module {}'.format(config[0]))
        raise Exception

    return module
