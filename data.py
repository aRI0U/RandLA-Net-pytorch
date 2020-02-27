import glob
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from utils.ply import read_ply

class PointCloudsDataset(Dataset):
    def __init__(self, dir, train=False, is_cuda=False):
        self.paths = glob.glob(dir+'/*.ply')
        self.size = len(self.paths)
        self.train = train
        self.is_cuda = is_cuda

    def __getitem__(self, idx):
        idx = idx % self.size
        path = self.paths[idx]
        points, labels = self.load_ply(path, keep_zeros=not self.train)

        points_tensor = torch.from_numpy(points)

        labels_tensor = torch.from_numpy(labels).long() - 1 if self.train else None

        if self.is_cuda:
            points_tensor = points_tensor.cuda()
            labels_tensor = labels_tensor.cuda()

        return points_tensor, labels_tensor

    def __len__(self):
        return self.size

    @staticmethod
    def load_ply(path, keep_zeros=True):
        cloud_ply = read_ply(path)
        points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        labels = None
        if not keep_zeros:
            labels = cloud_ply['class']

            # balance training set
            points_list, labels_list = [], []
            for i in range(1, len(np.unique(labels))):
                try:
                    idx = np.random.choice(len(labels[labels==i]), 10000)
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

def data_loader(dir, train=False, is_cuda=False, **kwargs):
    dataset = PointCloudsDataset(dir, train, is_cuda)
    return DataLoader(dataset, batch_size=None, **kwargs)
