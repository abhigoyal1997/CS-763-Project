import torch
import torch.nn as nn

from src.modules import create_module


class AttnModel(nn.Module):
	def __init__(self, config, args):
		super(AttnModel, self).__init__()
		self.config = config

		self.in_features = args['in_features']

		self.layers = nn.ModuleList()
		i = 1
		while i < len(config):
			self.layers.append(create_module(config[i], self.in_features))
			self.layers[-1].eval()
			i += 1

		self.is_cuda = args['cuda']

	def forward(self, im, vq, debug=False):
		if debug:
			outputs = []

		for layer in self.layers:
			p = layer(im, vq).expand_as(im)
			vq = (p*im).sum(1) + vq
			if debug:
				outputs.append(vq)

		if debug:
			return vq, outputs
		else:
			return vq
