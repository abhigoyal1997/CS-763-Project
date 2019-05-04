import torch
import torch.nn as nn

from src.modules import create_module
from tqdm import tqdm as tqdm


class VQAModel(nn.Module):
	def __init__(self, config, args):
		super(VQAModel, self).__init__()

		self.config = config
		self.im = args['im']
		self.tm = args['tm']

		self.layers = nn.ModuleList()
		i = 3
		in_features = self.im.out_shape
		x = torch.empty(2,in_features)
		if args['cuda']:
			x = x.cuda()

		while i < len(config):
			self.layers.append(create_module(config[i], x.size(1)))
			self.layers[-1].eval()
			with torch.no_grad():
				x = self.layers[-1](x)
			i += 1

		self.out_shape = x.size(1)
		self.is_cuda = args['cuda']

	def forward(self, x, q, lengths, debug=False):
		if debug:
			x, outputs = self.im(x, debug=True)
		else:
			x = self.im(x)

		if debug:
			q, outputs = self.tm(q, lengths, debug=True)
		else:
			x = self.tm(q, lengths, x.unsqueeze(0), x.unsqueeze(0))

		# x = torch.cat([x,q], dim=1)
		# x = x*q

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

	def cuda(self, device=None):
		self.is_cuda = True
		self.tm.is_cuda = True
		self.im.is_cuda = True
		return super(VQAModel, self).cuda(device)

	def get_criterion(self):
		if self.out_shape == 1:
			bce_loss = nn.BCEWithLogitsLoss()

			def loss(logits, y):
				return bce_loss(logits.expand_as(y), y)

			return loss
		else:
			pass  # TODO: handle other cases (numeric answers and open ended answers)

	def run_epoch(self, mode, batches, epoch, criterion=None, optimizer=None, writer=None, log_interval=None):
		if mode == 'train':
			self.train()
		else:
			self.eval()

		loss = 0.0
		acc = 0
		i = 0
		data_size = 0
		for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
			if self.is_cuda:
				for k in range(len(data)):
					data[k] = data[k].cuda()
			x,q,lengths,y = sort_batch(*data)

			with torch.set_grad_enabled(self.training):
				if self.training:
					optimizer.zero_grad()

				# Forward Pass
				logits = self.forward(x,q,lengths)
				batch_loss = criterion(logits, y)

				if self.training:
					# Backward Pass
					batch_loss.backward()
					optimizer.step()

			# Update metrics
			loss += batch_loss.item()*x.shape[0]
			if self.out_shape == 1:
				predictions = (torch.sigmoid(logits) > 0.5).float()
			else:
				# TODO: handle other cases
				pass

			acc += torch.clamp(((predictions.expand_as(y) == y).sum(dim=1,keepdim=True).expand_as(y) - (predictions.expand_as(y) == y).long()).float()/3, 0.0, 1.0).mean(dim=1).sum(dim=0).item()
			data_size += x.shape[0]

			if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
				writer.add_scalar('{}_loss'.format(mode), loss/data_size, epoch*len(batches)+i)
			i += 1

		loss /= data_size
		accuracy = acc/data_size
		if writer is not None:
			writer.add_scalar('{}_acc'.format(mode), accuracy, epoch)
			if mode == 'valid':
				writer.add_scalar('{}_loss'.format(mode), loss, epoch)
		return {'loss': loss, 'acc': accuracy}

	def predict(self, batches, labels=True):
		self.eval()
		predictions = None
		if labels:
			y_true = None
		with torch.no_grad():
			for data in tqdm(batches):
				if self.is_cuda:
					for k in range(len(data)):
						data[k] = data[k].cuda()
				if labels:
					x,q,lengths,y = sort_batch(*data)
				else:
					x,q,lengths = sort_batch(*data)

				# Forward Pass
				logits = self.forward(x,q,lengths)

				# Update metrics
				if predictions is None:
					if self.out_shape == 1:
						predictions = (torch.sigmoid(logits) > 0.5).long()
					else:
						# TODO: handle other cases
						pass
				else:
					predictions = torch.cat([predictions, torch.argmax(logits,dim=1)])
					if labels:
						y_true = torch.cat([y_true, y])

			if labels:
				accuracy = (predictions == y_true.long()).double().mean().item()
				return {'predictions': predictions, 'acc': accuracy}
			else:
				return {'predictions': predictions}


def sort_batch(x, q, lengths, y=None):
	lengths, indx = lengths.sort(dim=0, descending=True)
	x = x[indx]
	q = q[indx]
	if y is not None:
		y = y[indx]
		return x, q.transpose(0,1), lengths, y
	else:
		return x, q.transpose(0,1), lengths
