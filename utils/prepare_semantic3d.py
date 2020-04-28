import numpy as np
from pathlib import Path
from tqdm import tqdm

from ply import write_ply

output_type = 'npy'
i_dont_have_a_lot_of_memory_ok = True
ROOT_PATH = (Path(__file__) / '..' / '..').resolve()
DATASET_PATH = ROOT_PATH / 'semantic3d'
RAW_PATH = DATASET_PATH / 'original_data'
TRAIN_PATH = DATASET_PATH / 'train'
TEST_PATH = DATASET_PATH / 'test'
VAL_PATH = DATASET_PATH / 'val'

for folder in [TRAIN_PATH, TEST_PATH, VAL_PATH]:
    folder.mkdir(exist_ok=True)

print(f'Computing point clouds as {output_type} files. This operation is very time-consuming.')

for pc_path in RAW_PATH.glob('*.txt'):
    name = pc_path.stem

    if output_type=='ply' :
        pc_name = name + '.ply'
    elif output_type=='npy':
        pc_name = name + '.npy'
    else :
        raise 'unknown output_type'

    # Check if the point cloud has already been computed
    if list(ROOT_PATH.rglob(pc_name)) != []:
        continue

    print(f'Writing {pc_name}...')

    # In order to avoid memory over flow
    if i_dont_have_a_lot_of_memory_ok :

        labels_path = RAW_PATH / (name + '.labels')
        training_data = False
        dir = TEST_PATH
        if labels_path.exists() :
            labels = np.loadtxt(labels_path, dtype=np.float32)
            training_data = True
            dir = VAL_PATH if '3' in name else TRAIN_PATH
        bunchsize = 1e10^5
        index = 0
        points = []

        with open(pc_path, 'r') as file :
            for line in tqdm(file):
                points.append(np.loadtxt(line, dtype=np.float32))
                if len(points) == bunchsize:
                    points = np.array(points)
                    prev_data = np.load(dir / pc_name)
                    if training_data :
                        np.save(dir / pc_name, np.concatenate((prev_data, np.vstack((points.T, labels[index*bunchsize:(index+1)*bunchsize])).T)))
                    else :
                        np.save(dir / pc_name, np.concatenate((prev_data, points)))
                    points = []
                    index += 1
            points = np.array(points)
            prev_data = np.load(dir / pc_name)
            if training_data :
                np.save(dir / pc_name, np.concatenate((prev_data, np.vstack((points.T, labels[index*bunchsize:])).T)))
            else :
                np.save(dir / pc_name, np.concatenate((prev_data, points)))

    # If you are memory rich
    else :
        try:
            points = np.loadtxt(pc_path, dtype=np.float32)
        except:
            continue

        labels_path = RAW_PATH / (name + '.labels')
        if labels_path.exists():
            labels = np.loadtxt(labels_path, dtype=np.float32)
            dir = VAL_PATH if '3' in name else TRAIN_PATH
            if output_type=='ply' :
                write_ply(dir / pc_name, [points, labels], 'x y z intensity red green blue class'.split(' '))
            else :
                np.save(dir / pc_name, np.vstack((points.T, labels)).T)
            print(f'As {dir / pc_name}...')

        else:
            if output_type=='ply' :
                write_ply(TEST_PATH / pc_name, points, 'x y z intensity red green blue'.split(' '))
            else :
                np.save(TEST_PATH / pc_name, points)

print('Done.')
