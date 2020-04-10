import numpy as np
from pathlib import Path
import time

import torch
import torch.nn as nn

from data import data_loader
from model import RandLANet
from utils.ply import read_ply, write_ply

t0 = time.time()

path = Path('datasets') / 's3dis' / 'reprocessed' / 'test'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Loading data...')
loader = data_loader(path, device, train=False)

print('Loading model...')

d_in = 6
num_classes = 14

model = RandLANet(d_in, num_classes, 16, 4, device)
model.load_state_dict(torch.load('runs/2020-04-10_20:15/checkpoint_20.pth')['model_state_dict'])
model.eval()

points, _ = next(iter(loader))

print('Predicting labels...')
with torch.no_grad():
    print(points.shape)
    scores = model(points)
    print(scores.shape)
    predictions = (torch.max(scores, dim=-1).indices).cpu().numpy().astype(np.int32)

print('Writing results...')
np.savetxt('output.txt', predictions, fmt='%d', delimiter='\n')

t1 = time.time()
# write point cloud with classes
print('Assigning labels to the point cloud...')
cloud = points.squeeze(0)[:,:3]
write_ply('MiniDijon9.ply', [cloud, predictions], ['x', 'y', 'z', 'class'])

print('Done. Time elapsed: {:.1f}s'.format(t1-t0))
