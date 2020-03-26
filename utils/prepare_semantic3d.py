import numpy as np
from pathlib import Path
from tqdm import tqdm

from ply import write_ply

ROOT_PATH = (Path(__file__) / '..' / '..').resolve()
DATASET_PATH = ROOT_PATH / 'data' / 'semantic3d'
RAW_PATH = DATASET_PATH / 'original_data'
# TRAIN_PATH = ROOT_PATH / 'train'
# TEST_PATH = ROOT_PATH / 'test'
# VAL_PATH = ROOT_PATH / 'val'
print('Computing point clouds as ply files. This operation is very time-consuming.')

for pc_path in RAW_PATH.glob('*.txt'):
    name = pc_path.stem
    pc_name = name + '.ply'

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
        write_ply(DATASET_PATH / 'train' / pc_name, [points, labels], 'x y z intensity red green blue class'.split(' '))

    else:
        write_ply(DATASET_PATH / 'test' / pc_name, points, 'x y z intensity red green blue'.split(' '))

print('Done.')
