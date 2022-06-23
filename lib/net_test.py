from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
import pandas as pd
import numpy as np
import torch


import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super(PointNet, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')  # out: 32 ->m.p. 16
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 16
        self.conv_0_1 = nn.Conv3d(32, hidden_dim, 3, padding=1, padding_mode='replicate')  # out: 16 ->m.p. 8
        self.conv_1 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='replicate')  # out: 8
        self.conv_1_1 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='replicate')  # out: 8

        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(hidden_dim)
        self.conv1_1_bn = nn.BatchNorm3d(hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        net = self.maxpool(net)  # out 16

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        net = self.maxpool(net)  # out 8

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        fea = self.conv1_1_bn(net)

        return fea


def get_pcd():
    plydata = PlyData.read('/media/mana/mana/dataset/Titmouse_pifu/PCD/0000/titmouse_0000_watertight_pred.ply')
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=float)
    property_names = data[0].dtype.names
    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]
    pcd_data = data_np[:, :3]
    return torch.tensor(pcd_data).float()


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


if __name__ == '__main__':


    # grid_points = create_grid_points_from_bounds(-1.0, 1.0, 32)
    # kdtree = KDTree(grid_points)
    # pcd = get_pcd().unsqueeze(0).permute(0, 2, 1)
    #
    # occupancies = pcd.new_zeros(pcd.size(0), 32)
    # kp_pred = pcd.transpose(1, 2).detach().cpu().numpy()
    #
    # for b in range(pcd.size(0)):
    #     _, idx = kdtree.query(kp_pred[b])
    #     occupancies[b, idx] = 1
    #
    # voxel_kp_pred = occupancies.view(pcd.size(0), 32, 32, 32)
    # print(voxel_kp_pred)

    net = PointNet()
    print(net)
