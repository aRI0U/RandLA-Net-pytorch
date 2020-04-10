import glob, pickle, time, warnings
import numpy as np
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

from config import cfg
from utils.ply import read_ply
from utils.tools import DataProcessing as DP

class PointCloudsDataset(Dataset):
    def __init__(self, dir, device, train=False, data_type='npy'):
        self.paths = list(dir.glob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.train = train
        self.device = device
        self.data_type = data_type

    def __getitem__(self, idx):
        idx = idx % self.size
        path = self.paths[idx]

        if self.data_type=='npy':
            points, labels = self.load_npy(path, keep_zeros=not self.train)
        elif self.data_type=='ply':
            points, labels = self.load_ply(path, keep_zeros=not self.train)
        else :
            raise 'unknown data type, compatible types are "npy" (preferred) and "ply" point clouds'

        points_tensor = torch.from_numpy(points).float().to(self.device)
        labels_tensor = torch.from_numpy(labels).long().to(self.device) - 1

        # print(points_tensor.dtype, labels_tensor.dtype)
        return points_tensor, labels_tensor

    def __len__(self):
        return self.size

    @staticmethod
    def load_npy(path, keep_zeros=True):
        r"""
            load the point cloud and labels of the npy file located in path

            Args:
                path: str
                    path of the point cloud
                keep_zeros: bool (optional)
                    keep unclassified points
        """
        cloud_npy = np.load(path, mmap_mode='r')
        # print(cloud_npy.shape)
        points = cloud_npy[:, :6]

        labels = np.zeros(1)
        if not keep_zeros:
            labels = cloud_npy[:,-1]

            # balance training set
            points_list, labels_list = [], []
            for i in range(1, len(np.unique(labels))):
                try:
                    idx = np.random.choice(len(labels[labels==i]), 8000)
                    points_list.append(points[labels==i][idx])
                    labels_list.append(labels[labels==i][idx])
                except ValueError:
                    continue
            if points_list :
                points = np.stack(points_list)
                labels = np.stack(labels_list)
                labeled = labels>0
                points = points[labeled]
                labels = labels[labeled]

        return points, labels

    @staticmethod
    def load_ply(path, keep_zeros=True):
        r"""
            load the point cloud and labels of the ply file located in path

            Args:
                path: str
                    path of the point cloud
                keep_zeros: bool (optional)
                    keep unclassified points
        """
        cloud_ply = read_ply(path)
        points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        labels = None
        if not keep_zeros:
            labels = cloud_ply['class']

            # balance training set
            points_list, labels_list = [], []
            for i in range(1, len(np.unique(labels))):
                try:
                    idx = np.random.choice(len(labels[labels==i]), 8000)
                    points_list.append(points[labels==i][idx])
                    labels_list.append(labels[labels==i][idx])
                except ValueError:
                    continue

            points = np.stack(points_list)
            labels = np.stack(labels_list)
            labeled = labels>0
            points = points[labeled]
            labels = labels[labeled]

        return points, labels


class CloudsDataset(Dataset):
    def __init__(self, dir, device, train=False, data_type='npy'):
        self.path = dir
        self.paths = list(dir.glob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.device = device
        self.train = train
        self.data_type = data_type
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.val_split = '1_'
        self.val_proj = []
        self.val_labels = []

        self.load_data()
        print('Size of training : ', len(self.input_colors['training']))
        print('Size of validation : ', len(self.input_colors['validation']))

    def load_data(self):
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(self.path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_npy_file = join(self.path, '{:s}.npy'.format(cloud_name))

            data = np.load(sub_npy_file, mmap_mode='r').T
            sub_colors = data[:,3:6]
            sub_labels = data[:,-1].copy()

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # The points information is in tree.data
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(self.path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size


class active_learning_batch_sampler(BatchSampler):

    def __init__(self, dataset, device, split='training'):
        self.dataset = dataset
        self.batch_size = 6
        self.split = split
        self.possibility = {}
        self.min_possibility = {}
        self.device = device

        #Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def spatially_regular_gen(self):

        if self.split == 'training':
            num_per_epoch = 50 #cfg.train_steps * cfg.batch_size
        elif self.split == 'validation':
            num_per_epoch = 10 #cfg.val_steps * cfg.val_batch_size
        # Generator loop
        for i in range(num_per_epoch):  # num_per_epoch
            # t0 = time.time()

            # Choose a random cloud
            cloud_idx = int(np.argmin(self.min_possibility[self.split]))

            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility[self.split][cloud_idx])

            # Get points from tree structure
            points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            if len(points) < cfg.num_points:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
            else:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

            queried_idx = DP.shuffle_idx(queried_idx)
            # Collect points and colors
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
            queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[self.split][cloud_idx][queried_idx] += delta
            self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

            if len(points) < cfg.num_points:
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                    DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz.astype(np.float32)).to(self.device)
            queried_pc_colors = torch.from_numpy(queried_pc_colors.astype(np.float32)).to(self.device)
            queried_pc_labels = torch.from_numpy(queried_pc_labels.astype(np.float32)).to(self.device)
            queried_idx = torch.from_numpy(queried_idx.astype(np.float32)).to(self.device)
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32).astype(np.float32)).to(self.device)

            points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)
            # print('time in seconds : ', time.time() - t0)

            yield ( torch.reshape(points, (1, cfg.num_points, 6)),  torch.reshape(queried_pc_labels, (1, cfg.num_points)).long())

# def data_loader(dir, device, train=False, split='training', **kwargs):
#     dataset = PointCloudsDataset(dir, device, train)
#     return DataLoader(dataset, **kwargs)

def data_loader(dir,  device, train=False, split='training', **kwargs):
    dataset = CloudsDataset(dir, device, train)
    batch_sampler = active_learning_batch_sampler(dataset, device, split=split)
    return batch_sampler

if __name__ == '__main__':
    dataset = CloudsDataset('datasets/s3dis/subsampled/train')
    batch_sampler = active_learning_batch_sampler(dataset)
    for data in batch_sampler:
        xyz, colors, labbels, idx, cloud_idx = data
        print('Nomber of points : ', len(xyz))
        print('Point position : ', xyz[1])
        print('Color : ', colors[1])
        print('Label : ', labbels[1])
        print('Indice of point : ', idx[1])
        print('Cloud indice : ', cloud_idx)
        break
