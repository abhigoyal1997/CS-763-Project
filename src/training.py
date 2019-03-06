from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter
from time import time
from src.model_utils import save_model


def train(model, hparams, dataset, model_path=None, log_interval=None):
    batch_size = hparams['batch_size']
    num_epochs = hparams['num_epochs']
    train_ratio = hparams['train_ratio']
    num_workers = hparams['num_workers']

    train_size = int(train_ratio*len(dataset))
    train_set = Subset(dataset, list(range(train_size)))
    valid_set = Subset(dataset, list(range(train_size, len(dataset))))

    train_batches = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_batches = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)

    criterion = model.get_criterion()
    optimizer = Adam(model.parameters())

    log_path = model_path.replace('models','logs')+'_'+str(time())
    writer = SummaryWriter(log_dir=log_path)

    best_val = None
    for epoch in range(num_epochs):
        # Train
        metrics = model.run_epoch('train', train_batches, criterion=criterion, optimizer=optimizer, epoch=epoch, writer=writer, log_interval=log_interval)
        print('Train: {}'.format(metrics))

        # Validate
        metrics = model.run_epoch('valid', valid_batches, criterion=criterion, epoch=epoch, writer=writer, log_interval=log_interval)
        print('Validation: {}'.format(metrics))

        if 'acc' in metrics:
            if best_val is None or metrics['acc'] > best_val:
                best_val = metrics['acc']
                save_model(model, model_path)
        elif 'score' in metrics:
            if best_val is None or metrics['score'] > best_val:
                best_val = metrics['score']
                save_model(model, model_path)
        elif 'dice' in metrics:
            if best_val is None or metrics['dice'] > best_val:
                best_val = metrics['dice']
                save_model(model, model_path)
        elif 'mse' in metrics:
            if best_val is None or metrics['mse'] < best_val:
                best_val = metrics['mse']
                save_model(model, model_path)
