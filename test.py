import os
import time
import numpy as np

import torch
import torch.nn as nn

from data import data_loader
from model import RandLANet
from utils.ply import read_ply, write_ply

t0 = time.time()

path = os.path.join('data', 'test')

is_cuda = False#torch.cuda.is_available()

print('Loading data...')
loader = data_loader(path, train=False, is_cuda=is_cuda)

print('Loading model...')
model = RandLANet(6, 16, 4)
model.load_state_dict(torch.load('runs/2020-02-27_18:42/checkpoint_20.pth'))
model.eval()

points, _ = next(iter(loader))

print('Predicting labels...')
with torch.no_grad():
    scores = model(points)
    print(scores)
    predictions = (torch.max(scores, dim=-1).indices + 1).cpu().numpy().astype(np.int32)

print('Writing results...')
np.savetxt('MiniDijon9.txt', predictions, fmt='%d')

t1 = time.time()
# write point cloud with classes
print('Assigning labels to the point cloud...')
cloud_ply = read_ply('data/test/MiniDijon9.ply')
cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
write_ply('MiniDijon9.ply', [cloud, predictions], ['x', 'y', 'z', 'class'])

print('Done. Time elapsed: {:.1f}s'.format(t1-t0))
