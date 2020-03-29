import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

from ply import write_ply

output_type = 'npy'
ROOT_PATH = (Path(__file__) / '..' / '..').resolve()
DATASET_PATH = ROOT_PATH / 'data' / 'semantic3d'
RAW_PATH = DATASET_PATH / 'original_data'
TRAIN_PATH = DATASET_PATH / 'train'
TEST_PATH = DATASET_PATH / 'test'
VAL_PATH = DATASET_PATH / 'val'

for folder in [TRAIN_PATH, TEST_PATH, VAL_PATH] :
    if not os.path.exists(folder) :
        os.makedirs(folder)

print('Computing point clouds as ply files. This operation is very time-consuming.')

for pc_path in RAW_PATH.glob('*.txt'):
    name = pc_path.stem

    if output_type=='ply' :
        pc_name = name + '.ply'
    elif output_type=='npy' :
        pc_name = name + '.npy'
    else :
        raise 'unknown output_type'

    if list(ROOT_PATH.rglob(pc_name)) != []:
        continue

    print(f'Writing {pc_name}...')
    try:
        points = np.loadtxt(pc_path, dtype=np.float32)
    except:
        continue

    labels_path = RAW_PATH / (name + '.labels')
    if labels_path.exists():
        labels = np.loadtxt(labels_path)
        dir = 'val' if '3' in name else 'train'
        if output_type=='ply' :
            write_ply(DATASET_PATH / 'train' / pc_name, [points, labels], 'x y z intensity red green blue class'.split(' '))
        else :
            np.save(DATASET_PATH / 'train' / pc_name, np.vstack((points.T, labels)).T)

    else:
        if output_type=='ply' :
            write_ply(DATASET_PATH / 'test' / pc_name, points, 'x y z intensity red green blue'.split(' '))
        else :
            np.save(DATASET_PATH / 'test' / pc_name, points)

print('Done.')
