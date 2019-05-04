import torch
import torch.nn as nn

from src.modules import create_module


class TextModel(nn.Module):
	def __init__(self, config, args):
		super(TextModel, self).__init__()

		self.config = config

		self.embed = create_module(config[1], args['vocab_size'])
		self.vocab_size = self.embed.num_embeddings
		self.rcell = create_module(config[2], self.embed.embedding_dim)

		x = torch.empty(2, config[2][1])
		if args['cuda']:
			x = x.cuda()
		self.layers = nn.ModuleList()
		i = 3
		while i < len(config):
			self.layers.append(create_module(config[i], x.size(1)))
			self.layers[-1].eval()
			with torch.no_grad():
				x = self.layers[-1](x)
			i += 1

		self.out_shape = x.size(1)
		self.is_cuda = args['cuda']

	def cuda(self, device=None):
		self.is_cuda = True
		return super(TextModel, self).cuda(device)

	def forward(self, x, lengths, h0=None, c0=None, debug=False):
		if debug:
			outputs = []

		x = self.embed(x)
		if debug:
			outputs.append(x)

		x = nn.utils.rnn.pack_padded_sequence(x,lengths)
		if h0 is None:
			_, (h,_) = self.rcell(x)
		else:
			_, (h,_) = self.rcell(x, (h0,c0))
		x = h[-1]

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
