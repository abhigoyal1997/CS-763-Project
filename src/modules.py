import torch
import torch.nn as nn
import numpy as np
from torchvision import models


class Flat(nn.Module):
	def __init__(self):
		super(Flat, self).__init__()

	def forward(self, x):
		return x.view(x.size(0),-1)


class Normalize(nn.Module):
	def __init__(self, mean=None, std=None):
		super(Normalize, self).__init__()
		self.mean = mean
		self.std = std
		if self.mean is None:
			self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
		if self.std is None:
			self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

	def forward(self, x):
		if x.size(1) == 1:
			x = torch.cat([x]*3, dim=1)
		return (x-self.mean)/self.std

	def cuda(self, device=None):
		self.mean = self.mean.cuda()
		self.std = self.std.cuda()
		return super(Normalize, self).cuda(device)


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
		'pad2d': nn.ZeroPad2d,
		'normalize': Normalize,
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


def create_module(config, in_features=None, cuda=True):
	if config[0] == 'embed':
		if config[1] == 'p':
			module = pretrained_embedding_layer(config[2:])
		else:
			module = nn.Embedding(in_features, *config[1:])
	elif config[0] == 'vgg16':
		module = models.vgg16(True)
		module.classifier[-1] = nn.Linear(4096, int(config[1]))
	elif config[0] == 'vgg16_bn':
		module = models.vgg16_bn(True)
		module.classifier[-1] = nn.Linear(4096, int(config[1]))
	else:
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

	if cuda:
		return module.cuda()

	return module
