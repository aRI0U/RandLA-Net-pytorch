import torch
import torch.nn as nn
# from torch_geometric.nn import knn
from torch_points import knn

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        K = self.num_neighbors
        # finding neighboring points
        print(coords.cpu().shape)
        idx, dist = knn(coords.cpu(), coords.cpu(), K)
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(-3).repeat(1, 3, 1, 1)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).repeat(1, 1, 1, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3)

        return torch.cat((
            self.mlp(concat),
            features.repeat(1, 1, 1, K)
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """

        x = self.mlp1(features)

        x = self.lse1(coords, x)
        x = self.pool1(x)

        x = self.lse2(coords, x)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class RandLANet(nn.Module):
    def __init__(self, num_classes, num_neighbors, decimation):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(3, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoding = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors),
            LocalFeatureAggregation(32, 64, num_neighbors),
            LocalFeatureAggregation(128, 128, num_neighbors),
            LocalFeatureAggregation(256, 256, num_neighbors)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoding_channels = [512, 256, 128, 32, 8]
        self.decoding = nn.ModuleList([
            SharedMLP(
                decoding_channels[i],
                decoding_channels[i+1],
                transpose=True,
                bn=True,
                activation_fn=nn.ReLU()
            )
            for i in range(len(decoding_channels)-1)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d)
                input points

            Returns
            -------
            torch.Tensor, shape (B, N, num_classes)
                segmentation scores for each point
        """
        N = input.size(-2)
        d = self.decimation
        # print('input')
        # print(input, input.shape)
        coords = input[...,:3].clone()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)
        # print('fc_start')
        # print(x, x.shape)
        decimation_ratio = 1

        coords_saved = coords.clone()

        # <<<<<<<<<< ENCODER
        # coords, shape (B, N, 3)
        # x, shape (B, d, N, 1)
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
            coords, x = coords[:,idx], x[...,idx]
        # >>>>>>>>>> ENCODER

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
            neighbors, _ = knn(coords, new_coords, 1)
            x = torch.cat((x[neighbors], x_stack.pop()), dim=-1)
            # print('dec')
            # print(x, x.shape)
            x = mlp(x)
            # print('decoding')
            # print(x, x.shape)

        # print('\nDone.')
        scores = self.fc_end(x)
        return scores.squeeze(-1).transpose(-2,-1)





if __name__ == '__main__':
    import time
    cloud = 1000*torch.randn(1, 2**10, 3)
    model = RandLANet(6, 16, 4)
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
