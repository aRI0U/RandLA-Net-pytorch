import argparse
from datetime import datetime
from numpy import mean
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data import data_loader
from model import RandLANet


def train(args):
    path = os.path.join(args.dataset, args.train_dir)
    val_path = os.path.join(args.dataset, args.val_dir)
    logs_dir = os.path.join(args.logs_dir, args.name)
    os.makedirs(logs_dir, exist_ok=True)

    loader = data_loader(path, train=True, is_cuda=args.gpu, batch_size=args.batch_size)
    validation_loader = data_loader(val_path, train=True, is_cuda=args.gpu, batch_size=args.batch_size)
    # d_in = next(iter(loader)).size(-1)

    model = RandLANet(7, args.n_classes, args.neighbors, args.decimation)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)

    if args.gpu:
        model = model.cuda()

    with SummaryWriter(logs_dir) as writer:
        for epoch in range(1, args.epochs+1):

            # Train
            model.train()
            losses = []
            for points, labels in loader:
                optimizer.zero_grad()

                pred = model(points)

                loss = criterion(pred.squeeze(), labels.squeeze())

                loss.backward()

                optimizer.step()
                scheduler.step()

                losses.append(loss.cpu().item())

            print('[Epoch {:d}/{:d}]\tLoss: {:.7f}'.format(epoch, args.epochs, mean(losses)))
            writer.add_scalar('Training loss', mean(losses), epoch)

            model.eval()

            # Validation
            val_losses = []
            for points, labels in validation_loader :

                pred = model(points)

                loss = criterion(pred.squeeze(), labels.squeeze())

                val_losses.append(loss.cpu().item())

            print('[Validation : epoch {:d}/{:d}]\tLoss: {:.7f}'.format(epoch, args.epochs, mean(val_losses)))
            writer.add_scalar('Validation loss', mean(val_losses), epoch)

            if epoch % 10 == 0:
                torch.save(model.state_dict(), 'runs/{}/checkpoint_{:d}.pth'.format(args.name, epoch))


if __name__ == '__main__':

    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    base.add_argument('--dataset', type=str, help='location of the dataset',
                        default='/media/tibo/Maxtor/Data/Deepdata/points_cloud/semantic3d')

    expr.add_argument('--epochs', type=int, help='number of epochs',
                        default=20)
    expr.add_argument('--n_classes', type=int, help='number of classes',
                        default=8)

    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                        default=1e-2)
    param.add_argument('--batch_size', type=int, help='batch size',
                        default=1)
    param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                        default=4)
    param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                        default=16)
    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                        default=0.99)

    dirs.add_argument('--test_dir', type=str, help='location of the test set in the dataset dir',
                        default='test')
    dirs.add_argument('--train_dir', type=str, help='location of the training set in the dataset dir',
                        default='train')
    dirs.add_argument('--val_dir', type=str, help='location of the validation set in the dataset dir',
                        default='val')
    dirs.add_argument('--logs_dir', type=str, help='path to tensorboard logs',
                        default='runs')

    misc.add_argument('--gpu', action='store_true', help='use CUDA on GPU')
    misc.add_argument('--name', type=str, help='name of the experiment',
                        default=None)

    args = parser.parse_args()

    args.gpu = args.gpu and torch.cuda.is_available()

    if args.name is None:
        args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

    t0 = time.time()
    train(args)
    t1 = time.time()

    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
