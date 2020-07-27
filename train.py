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
from utils.tools import Config as cfg
from utils.metrics import accuracy, intersection_over_union

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    accuracies = []
    ious = []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc='Validation', leave=False):
            points = points.to(device)
            labels = labels.to(device)
            scores = model(points)
            loss = criterion(scores, labels)
            losses.append(loss.cpu().item())
            accuracies.append(accuracy(scores, labels))
            ious.append(intersection_over_union(scores, labels))
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0)


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

    n_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device=args.gpu)
    ratio_samples = n_samples / n_samples.sum()
    weights = 1 / (ratio_samples + 0.02)

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
            print(f'=== EPOCH {epoch:d}/{args.epochs:d} ===')
            t0 = time.time()
            # Train
            model.train()

            # metrics
            losses = []
            accuracies = []
            ious = []

            # iterate over dataset
            for points, labels in tqdm(train_loader, desc='Training', leave=False):
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
                ious.append(intersection_over_union(scores, labels))

            scheduler.step()

            accs = np.nanmean(np.array(accuracies), axis=0)
            ious = np.nanmean(np.array(ious), axis=0)

            val_loss, val_accs, val_ious = evaluate(
                model,
                val_loader,
                criterion,
                args.gpu
            )

            loss_dict = {
                'Training loss':    np.mean(losses),
                'Validation loss':  val_loss
            }
            acc_dicts = [
                {
                    'Training accuracy': acc,
                    'Validation accuracy': val_acc
                } for acc, val_acc in zip(accs, val_accs)
            ]
            iou_dicts = [
                {
                    'Training accuracy': iou,
                    'Validation accuracy': val_iou
                } for iou, val_iou in zip(ious, val_ious)
            ]

            t1 = time.time()
            d = t1 - t0
            # Display results
            for k, v in loss_dict.items():
                print(f'{k}: {v:.7f}', end='\t')
            print()

            print('Accuracy     ', *[f'{i:>5d}' for i in range(num_classes)], '   OA', sep=' | ')
            print('Training:    ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in accs], sep=' | ')
            print('Validation:  ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in val_accs], sep=' | ')

            print('IoU          ', *[f'{i:>5d}' for i in range(num_classes)], ' mIoU', sep=' | ')
            print('Training:    ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in ious], sep=' | ')
            print('Validation:  ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in val_ious], sep=' | ')

            print('Time elapsed:', '{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60)))

            # send results to tensorboard
            writer.add_scalars('Loss', loss_dict, epoch)

            for i in range(num_classes):
                writer.add_scalars(f'Per-class accuracy/{i+1:02d}', acc_dicts[i], epoch)
                writer.add_scalars(f'Per-class IoU/{i+1:02d}', iou_dicts[i], epoch)
            writer.add_scalars('Per-class accuracy/Overall', acc_dicts[-1], epoch)
            writer.add_scalars('Per-class IoU/Mean IoU', iou_dicts[-1], epoch)

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
                        default=10)

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
