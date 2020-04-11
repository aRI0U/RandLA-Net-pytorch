import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data import data_loaders
from model import RandLANet
from utils.config import cfg

def accuracy(scores, labels, num_classes=14):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc

        Parameters
        ----------
        scores:

        labels:

        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    predictions = torch.max(scores, dim=1).indices
    # per_class_acc = []
    # labels = labels[0]
    # for lab in range(num_classes):
    #     idx = (labels == lab)
    #     if len(idx) == 0 or labels[idx].shape[0]==0:
    #         proportion = 0
    #     else :
    #         correct = (pred_labels[idx] == labels[idx]).float().mean()
    #     per_class_acc.append(proportion)
    # class_accuracies.append(per_class_acc)
    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask * label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies

def evaluate(model, loader, criterion, device, desc=None):
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc=desc, leave=False, ncols=80):
            points = points.to(device)
            labels = labels.to(device)
            scores = model(points)
            loss = criterion(scores, labels)
            accuracies.append(accuracy(scores, labels))
            losses.append(loss.cpu().item())
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0)


def train(args):
    train_path = args.dataset / args.train_dir
    val_path = args.dataset / args.val_dir
    logs_dir = args.logs_dir / args.name
    logs_dir.mkdir(exist_ok=True, parents=True)

    # determine number of classes
    try:
        with open(args.dataset / 'classes.json') as f:
            labels = json.load(f)
            num_classes = len(labels.keys())
    except FileNotFoundError:
        num_classes = int(input("Number of distinct classes in the dataset: "))

    train_loader, val_loader = data_loaders(
        args.dataset,
        args.dataset_sampling,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    d_in = next(iter(train_loader))[0].size(-1)

    model = RandLANet(
        d_in,
        num_classes,
        num_neighbors=args.neighbors,
        decimation=args.decimation,
        device=args.gpu
    )

    print('Computing weights...', end='\t')
    samples_per_class = np.array(cfg.class_weights)
    # weight = samples_per_class / float(sum(samples_per_class))
    # class_weights = 1 / (weight + 0.02)
    # effective = 1.0 - np.power(0.99, samples_per_class)
    # class_weights = (1 - 0.99) / effective
    # class_weights = class_weights / (np.sum(class_weights) * num_classes)
    # class_weights = class_weights / float(sum(class_weights))
    # weights = torch.tensor(class_weights).float().to(args.gpu)
    n_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device=args.gpu, requires_grad=False)
    ratio_samples = n_samples / n_samples.sum()
    weights = ratio_samples
    #weights = F.softmin(n_samples)
    # weights = (1/ratio_samples) / (1/ratio_samples).sum()

    print('Done.')
    print('Weights:', weights)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)

    first_epoch = 1
    if args.load:
        path = max(list((args.logs_dir / args.load).glob('*.pth')))
        print(f'Loading {path}...')
        checkpoint = torch.load(path)
        first_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    with SummaryWriter(logs_dir) as writer:
        for epoch in range(first_epoch, args.epochs+1):
            t0 = time.time()
            # Train
            model.train()
            losses = []
            accuracies = []
            class_accuracies = []
            for points, labels in tqdm(train_loader, desc=f'[Epoch {epoch:d}/{args.epochs:d}]\tTraining', leave=False, ncols=80):
                points = points.to(args.gpu)
                labels = labels.to(args.gpu)
                optimizer.zero_grad()

                scores = model(points)

                logp = torch.distributions.utils.probs_to_logits(scores, is_binary=False)
                loss = criterion(logp, labels)
                # logpy = torch.gather(logp, 1, labels)
                # loss = -(logpy).mean()

                loss.backward()

                optimizer.step()

                losses.append(loss.cpu().item())
                accuracies.append(accuracy(scores, labels))

            scheduler.step()

            accs = np.nanmean(np.array(accuracies), axis=0)

            val_loss, val_accs = evaluate(
                model,
                val_loader,
                criterion,
                args.gpu,
                desc=f'[Epoch {epoch:d}/{args.epochs:d}]\tValidation'
            )

            loss_dict = {
                'Training loss':    np.mean(losses),
                'Validation loss':  val_loss
            }
            accuracy_dict = {
                'Training accuracy':    accs[-1],
                'Validation accuracy':  val_accs[-1]
            }

            # acc_dicts = [
            #     {
            #         f'{i:02d}_train_acc':    acc,
            #         f'{}':  val_acc
            #     }
            #     for i, (acc, val_accs) in enumerate(zip(accs, val_accs))
            # ]

            t1 = time.time()
            # Display results
            print(f'[Epoch {epoch:d}/{args.epochs:d}]', end='\t')
            for k, v in loss_dict.items():
                print(f'{k}: {v:.7f}', end='\t')

            d = t1 - t0
            print('\tTime elapsed:', '{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:.0f} s'.format(*divmod(d, 60)))
            print('train acc:', *[f'{acc:.3f}' for acc in accs], sep='   ')
            print('val acc:  ', *[f'{acc:.3f}' for acc in val_accs], sep='   ')
            writer.add_scalars('Loss', loss_dict, epoch)
            writer.add_scalars('Accuracy', accuracy_dict, epoch)

            if epoch % args.save_freq == 0:
                torch.save(
                    dict(
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict()
                    ),
                    args.logs_dir / args.name / f'checkpoint_{epoch:02d}.pth'
                )


if __name__ == '__main__':

    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    base.add_argument('--dataset', type=Path, help='location of the dataset',
                        default='datasets/s3dis/subsampled')

    expr.add_argument('--epochs', type=int, help='number of epochs',
                        default=50)
    expr.add_argument('--load', type=str, help='model to load',
                        default='')

    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                        default=1e-2)
    param.add_argument('--batch_size', type=int, help='batch size',
                        default=1)
    param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                        default=4)
    param.add_argument('--dataset_sampling', type=str, help='how dataset is sampled',
                        default='active_learning', choices=['active_learning', 'naive'])
    param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                        default=16)
    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                        default=0.95)

    dirs.add_argument('--test_dir', type=str, help='location of the test set in the dataset dir',
                        default='test')
    dirs.add_argument('--train_dir', type=str, help='location of the training set in the dataset dir',
                        default='train')
    dirs.add_argument('--val_dir', type=str, help='location of the validation set in the dataset dir',
                        default='val')
    dirs.add_argument('--logs_dir', type=Path, help='path to tensorboard logs',
                        default='runs')

    misc.add_argument('--gpu', type=int, help='which GPU to use (-1 for CPU)',
                        default=0)
    misc.add_argument('--name', type=str, help='name of the experiment',
                        default=None)
    misc.add_argument('--num_workers', type=int, help='number of threads for loading data',
                        default=0)
    misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints',
                        default=1)

    args = parser.parse_args()

    if args.gpu >= 0:
        if torch.cuda.is_available():
            args.gpu = torch.device(f'cuda:{args.gpu:d}')
        else:
            warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
            args.gpu = torch.device('cpu')
    else:
        args.gpu = torch.device('cpu')

    if args.name is None:
        if args.load:
            args.name = args.load
        else:
            args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

    t0 = time.time()
    train(args)
    t1 = time.time()

    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
