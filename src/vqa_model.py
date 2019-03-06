import torch
import torch.nn as nn

from tqdm import tqdm
from src.modules import create_module


class VQAModel(nn.Module):
    def __init__(self, config, args):
        super(VQAModel, self).__init__()

        self.config = config
        self.im = args['im']
        self.tm = args['tm']

        self.layers = nn.ModuleList()
        i = 3
        in_features = self.im.out_shape
        x = torch.empty(1,in_features)
        while i < len(config):
            self.layers.append(create_module(config[i], x.size(1)))
            with torch.no_grad():
                x = self.layers[-1](x)
            i += 1

        self.out_shape = x.size(1)
        self.is_cuda = False

    def forward(self, x, q, lengths, debug=False):
        if debug:
            x, outputs = self.im(x, debug=True)
        else:
            x = self.im(x)

        if debug:
            q, outputs = self.tm(q, lengths, debug=True)
        else:
            q = self.tm(q, lengths)

        x = x*q
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
        return super(VQAModel, self).cuda(device)

    def get_criterion(self):
        if self.out_shape == 1:
            bce_loss = nn.BCEWithLogitsLoss()

            def criterion(logits, y):
                return bce_loss(logits.expand_as(y), y)

            return criterion
        else:
            pass  # TODO: handle other cases (numeric answers and open ended answers)

    def run_epoch(self, mode, batches, epoch, criterion=None, optimizer=None, writer=None, log_interval=None):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        loss = 0.0
        correct_predictions = 0
        data_size = 0
        i = 0
        for data in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
            if self.is_cuda:
                for k in range(len(data)):
                    data[k] = data[k].cuda()
            x,q,lengths,y = sort_batch(*data)

            if mode == 'train':
                optimizer.zero_grad()

                # Forward Pass
                logits = self.forward(x,q,lengths)
                batch_loss = criterion(logits, y)

                # Backward Pass
                batch_loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    # Forward Pass
                    logits = self.forward(x,q,lengths)
                    batch_loss = criterion(logits, y)

            # Update metrics
            loss += batch_loss.item()
            if self.out_shape == 1:
                predictions = (torch.sigmoid(logits) > 0.5).long()
            else:
                # TODO: handle other cases
                pass

            correct_predictions += (predictions == y.long()).sum().item()
            data_size += x.shape[0]

            if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
                writer.add_scalar('{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
            i += 1

        loss = loss/len(batches)
        accuracy = correct_predictions/data_size
        if writer is not None:
            writer.add_scalar('{}_acc'.format(mode), accuracy, epoch)
            if mode == 'valid':
                writer.add_scalar('{}_loss'.format(mode), loss, epoch)
        return {'loss': loss, 'acc': accuracy}

    def predict(self, batches, labels=True):
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
