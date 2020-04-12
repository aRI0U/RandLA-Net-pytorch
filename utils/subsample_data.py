
from glob import glob
import numpy as np
from pathlib import Path

from sklearn.neighbors import KDTree
import pickle

from tools import Config as cfg
from tools import DataProcessing as DP
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

ROOT_PATH = (Path(__file__)  / '..' / '..').resolve()
DATASET_PATH = ROOT_PATH / 'datasets' / 's3dis'
NEW_PATH = DATASET_PATH / 'subsampled'
LABELS_PATH = DATASET_PATH / 'classes.json'
TRAIN_PATH = 'train'
TEST_PATH = 'test'
VAL_PATH = 'val'

LABELS_AVAILABLE_IN_TEST_SET = True

sub_grid_size = cfg.sub_grid_size

for folder in [TRAIN_PATH, TEST_PATH, VAL_PATH]:
    (NEW_PATH / folder).mkdir(parents=True, exist_ok=True)

    for file in (DATASET_PATH / folder).glob('*.npy'):
        file_name = file.stem
        print(file.name, end=':\t')
        if (NEW_PATH / folder / (file_name+'.npy')).exists():
            print('Already subsampled.')
            continue

        data = np.load(file, mmap_mode='r')

        print('Loaded data of shape : ', np.shape(data))


        # For each point cloud, a sub sample of point will be used for the nearest neighbors and training
        sub_npy_file = NEW_PATH / folder / (file_name + '.npy')
        xyz = data[:,:3].astype(np.float32)
        colors = data[:,3:6].astype(np.uint8)

        if folder!=TEST_PATH or LABELS_AVAILABLE_IN_TEST_SET:
            labels = data[:,-1].astype(np.uint8)
            sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
            sub_colors = sub_colors / 255.0
            np.save(sub_npy_file, np.concatenate((sub_xyz, sub_colors, sub_labels), axis=1).T)

        else:
            sub_xyz, sub_colors = DP.grid_sub_sampling(xyz, colors, None, sub_grid_size)
            sub_colors = sub_colors / 255.0
            np.save(sub_npy_file, np.concatenate((sub_xyz, sub_colors), axis=1).T)

        # The search tree is the KD_tree saved for each point cloud
        search_tree = KDTree(sub_xyz)
        kd_tree_file = NEW_PATH / folder / (file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

            # Projection is the nearest points of the selected grid to each point of the cloud
            proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            proj_save = NEW_PATH / folder / (file_name + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)
