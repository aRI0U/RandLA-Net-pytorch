from datetime import datetime
from numpy import mean
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data import data_loader
from model import RandLANet

num_classes = 6

path = os.path.join('data', 'training')
os.makedirs('runs', exist_ok=True)

is_cuda = False#torch.cuda.is_available()

loader = data_loader(path, train=True, is_cuda=is_cuda)
model = RandLANet(6, 16, 4)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

if is_cuda:
    model = model.cuda()

num_epochs = 200

name = datetime.now().strftime('%Y-%m-%d_%H:%M')

with SummaryWriter('runs/'+name) as writer:
    for epoch in range(1, num_epochs+1):
        model.train()
        losses = []
        for points, labels in loader:
            optimizer.zero_grad()

            pred = model(points)

            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.cpu().item())

        print('[Epoch {:d}/{:d}]\tLoss: {:.7f}'.format(epoch, num_epochs, mean(losses)))
        writer.add_scalar('Training loss', mean(losses), epoch)

        model.eval()
        print(model(points))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'runs/{}/checkpoint_{:d}.pth'.format(name, epoch))
