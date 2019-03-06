import os
import sys
import torch

from src.image_model import ImageModel
from src.text_model import TextModel
from src.vqa_model import VQAModel

MODELS = {
    'i': ImageModel,
    't': TextModel
}


def save_model(model, model_path):
    with open(os.path.join(model_path,'config.txt'),'w') as f:
        f.write('\n'.join([' '.join([str(j) for j in i]) for i in model.config]))

    torch.save(model.state_dict(), os.path.join(model_path, '{}.pth'.format(model.config[0][0])))
    print('{} saved to {}'.format(model.config[0][0], model_path))


def read_config(config_file):
    def cast(x):
        try:
            if '.' in x:
                return float(x)
            else:
                return int(x)
        except Exception:
            return x

    with open(config_file, 'r') as f:
        config = f.readlines()
    for i in range(len(config)):
        config[i] = list(map(cast, config[i].split()))
    return config


def create_model(config, args=None, cuda=True):
    if config[0][0] == 'v':
        args['im'] = create_model(read_config(config[1][0]), args={'in_shape': args['image_shape']}, cuda=cuda)
        args['tm'] = create_model(read_config(config[2][0]), args={'vocab_size': args['vocab_size']}, cuda=cuda)
        model = VQAModel(config, args)
    elif config[0][0] in MODELS:
        model = MODELS[config[0][0]](config, args)
    else:
        print('{} not implemented!'.format(config[0][0]))
        sys.exit(0)

    if cuda:
        return model.cuda()
    else:
        return model


def load_model(model_path, cuda, weights=True):
    config = read_config(os.path.join(model_path, 'config.txt'))
    model = create_model(config, cuda)
    state_path = os.path.join(model_path, config[0][0]+'.pth')
    if os.path.exists(state_path) and weights:
        print('Loading weights from {}...'.format(state_path))
        if cuda:
            model.load_state_dict(torch.load(state_path, map_location='cuda'))
        else:
            model.load_state_dict(torch.load(state_path, map_location='cpu'))
    return model


def read_hparams(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    param_keys = [
        'batch_size',
        'num_epochs',
        'train_ratio',
        'num_workers'
    ]
    hparams = {}
    for i in range(len(param_keys)):
        if '.' in spec[i]:
            hparams[param_keys[i]] = float(spec[i])
        else:
            hparams[param_keys[i]] = int(spec[i])
    return hparams
