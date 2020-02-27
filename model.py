import torch
import torch.nn as nn
from torch_geometric.nn import knn

class SharedMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SharedMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)



class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp = SharedMLP(10, d//2, d)

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (N,3)
                coordinates of the point cloud
            features: torch.Tensor, shape (N,d)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (N,K,2*d)
        """
        # finding neighboring points
        original_idx, neighbor_idx = knn(coords.cpu(), coords.cpu(), self.num_neighbors)
        original, neighbor = coords[original_idx], coords[neighbor_idx]

        # relative point position encoding
        relative = original - neighbor
        distance = torch.norm(relative, dim=-1).unsqueeze(1)
        concat = torch.cat((original, neighbor, relative, distance), dim=-1)

        return torch.cat((
            self.mlp(concat),
            features[original_idx]
        ), dim=-1).view(coords.size(0), self.num_neighbors, -1)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.mlp = SharedMLP(in_channels, hidden_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # computing attention scores
        scores = self.softmax(x)
        # sum over the neighbors
        features = torch.sum(scores * x, dim=-2)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.mlp1 = SharedMLP(d_in, d_out//2, d_out//2)
        self.mlp2 = SharedMLP(d_out, 3*d_out//2, 2*d_out)
        self.mlp3 = SharedMLP(d_in, d_out, 2*d_out)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (N,3)
                coordinates of the point cloud
            features: torch.Tensor, shape (N,d_in)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (N,2*d_out)
        """

        x = self.mlp1(features)

        x = self.lse1(coords, x)
        x = self.pool1(x)

        x = self.lse2(coords, x)
        x = self.pool2(x)
        return self.lrelu(self.mlp2(x) + self.mlp3(features))



class RandLANet(nn.Module):
    def __init__(self, num_classes, num_neighbors, decimation):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(3, 8)

        # encoding layers
        self.encoding = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors),
            LocalFeatureAggregation(32, 64, num_neighbors),
            LocalFeatureAggregation(128, 128, num_neighbors),
            LocalFeatureAggregation(256, 256, num_neighbors)
        ])

        self.mlp = SharedMLP(512, 512, 512)

        # decoding layers
        self.decoding = nn.ModuleList([
            SharedMLP(1024, 512, 256),
            SharedMLP(512, 256, 128),
            SharedMLP(256, 128, 32),
            SharedMLP(64, 32, 8)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            nn.Linear(8, 64),
            nn.Linear(64, 32),
            nn.Dropout(),
            nn.Linear(32, num_classes)
        )

    def forward(self, input):
        N = input.size(0)
        d = self.decimation
        # print('input')
        # print(input, input.shape)
        coords, features = input, input
        x = self.fc_start(features)
        # print('fc_start')
        # print(x, x.shape)
        decimation_ratio = 1

        coords_saved = coords.clone()

        # print('Encoding the point cloud', end='', flush=True)
        idx_stack, x_stack = [], []
        idx = torch.arange(N)
        for lfa in self.encoding:
            # print('.', end='', flush=True)
            x = lfa(coords, x)
            # print('lfa')
            # print(x, x.shape)

            idx_stack.append(idx.clone())
            x_stack.append(x.clone())

            # random downsampling
            decimation_ratio *= d
            idx = torch.randperm(N*d//decimation_ratio)[:N//decimation_ratio]
            coords, x = coords[idx], x[idx]

        # print()

        x = self.mlp(x)
        # print('mlp')
        # print(x, x.shape)

        # print('Decoding the point cloud', end='', flush=True)
        for mlp in self.decoding:
            # print('.', end='', flush=True)
            # upsampling
            idx = idx_stack.pop()
            new_coords = coords_saved[idx]
            _, neighbors = knn(coords, new_coords, 1)
            x = torch.cat((x[neighbors], x_stack.pop()), dim=-1)
            # print('dec')
            # print(x, x.shape)
            x = mlp(x)
            # print('decoding')
            # print(x, x.shape)

        # print('\nDone.')
        return self.fc_end(x)





if __name__ == '__main__':
    import time
    cloud = 1000*torch.randn(2**20, 3)
    model = RandLANet(6, 4)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    model.eval()

    if False:#torch.cuda.is_available():
        model = model.cuda()
        cloud = cloud.cuda()

    t0 = time.time()
    pred = model(cloud)
    t1 = time.time()
    print(pred)
    print(t1-t0)
