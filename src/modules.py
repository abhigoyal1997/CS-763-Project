import torch.nn as nn


class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)


MODULES = {
    'nn': {
        'conv': nn.Conv2d,
        'linear': nn.Linear,
        'norm2d': nn.BatchNorm2d,
        'embed': nn.Embedding,
        'lstm': nn.LSTM,
        'rnn': nn.RNN
    },
    'f': {
        'max2d': nn.MaxPool2d,
        'relu': nn.ReLU,
        'drop1d': nn.Dropout,
        'drop2d': nn.Dropout2d,
        'flat': Flat,
    }
}


def create_module(config, in_features=None):
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
