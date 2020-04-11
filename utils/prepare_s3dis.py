from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import warnings

ROOT_PATH = (Path(__file__) / '..' / '..').resolve()
DATASET_PATH = ROOT_PATH / 'datasets' / 's3dis'
RAW_PATH = DATASET_PATH / 'Stanford3dDataset_v1.2'
LABELS_PATH = DATASET_PATH / 'classes.json'
TRAIN_PATH = DATASET_PATH / 'train'
TEST_PATH = DATASET_PATH / 'test'
VAL_PATH = DATASET_PATH / 'val'

for folder in [TRAIN_PATH, TEST_PATH, VAL_PATH]:
    folder.mkdir(exist_ok=True)


if LABELS_PATH.exists():
    print(LABELS_PATH)
    with open(LABELS_PATH, 'r') as f:
        labels_dict = defaultdict(lambda: len(labels_dict.keys()), json.load(f))
else:
    labels_dict = defaultdict(lambda: len(labels_dict.keys()))

for area_number in range(1,7):
    print(f'Reencoding point clouds of area {area_number:d}')
    dir = RAW_PATH / f'Area_{area_number:d}'
    if not dir.exists():
        warnings.warn(f'Area {area_number:d} not found')
        continue
    for pc_path in sorted(list(dir.iterdir())):
        if not pc_path.is_dir:
            continue
        pc_name = f'{area_number:d}_' + pc_path.stem + '.npy'

        # chack that the point cloud has not been traeated yet
        if list(ROOT_PATH.rglob(pc_name)) != []:
            continue

        points_list = []
        for elem in sorted(list(pc_path.glob('Annotations/*.txt'))):
            label = elem.stem.split('_')[0]
            print(f'Computation of {pc_name}: adding {label} to point cloud...          ', end='\r')
            points = np.loadtxt(elem, dtype=np.float32)
            label_id = labels_dict[label]
            labelled_points = np.vstack((points.T, np.full(points.shape[0], label_id))).T
            points_list.append(labelled_points.astype(np.float32))

        if points_list == []:
            continue
        # save dict as json
        with open(LABELS_PATH, 'w') as f:
            json.dump(labels_dict, f, indent=2)

        # merge all subclouds together
        merged_points = np.vstack(points_list)

        # save computed point cloud
        path = TRAIN_PATH if area_number < 5 else TEST_PATH
        np.save(path / pc_name, merged_points, allow_pickle=False)

print('Done.')
