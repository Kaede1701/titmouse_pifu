import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree as KDTree
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
from .VolumeFilters import VolumeEncoder
from .pointnet2 import PointNet2
from .PointNet import PointNet
from ..mesh_util import create_grid_points_from_bounds


class TitmouseNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.BCELoss(),
                 ):
        super(TitmouseNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'TitmouseNet'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        # 三维特征
        if self.opt.feat_type == 'point':
            self.point_filter = PointNet2()
        elif self.opt.feat_type == 'volume':
            self.volume_filter = VolumeEncoder(3, 32)
        elif self.opt.feat_type == 'point_voxel':
            self.point_voxel = PointNet(self.opt.pn_hid_dim)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        self.normalizer = DepthNormalizer(opt)

        # Point net
        self.reso_grid = self.opt.reso_grid

        bb_min = -1.0
        bb_max = 1.0

        self.grid_points = create_grid_points_from_bounds(bb_min, bb_max, self.reso_grid)
        self.kdtree = KDTree(self.grid_points)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.feat_grid_list = []

        self.intermediate_preds_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def pcd_filter(self, pcd):
        occupancies = pcd.new_zeros(pcd.size(0), len(self.grid_points))
        kp_pred = pcd.transpose(1, 2).detach().cpu().numpy()

        for b in range(pcd.size(0)):
            _, idx = self.kdtree.query(kp_pred[b])
            occupancies[b, idx] = 1

        voxel_kp_pred = occupancies.view(pcd.size(0), self.reso_grid, self.reso_grid, self.reso_grid)
        self.feat_grid_list = self.point_voxel(voxel_kp_pred.detach())

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        vgrid = xyz.transpose(1, 2)
        vgrid = vgrid[:, :, None, None, :]

        # new coding end

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        z_feat = self.normalizer(z, calibs=calibs)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        xyz_feat_grid_list = []
        for pcd_feat in self.feat_grid_list:
            xyz_feat_grid = F.grid_sample(pcd_feat, vgrid, padding_mode='border', align_corners=True,
                                          mode='bilinear').squeeze(-1).squeeze(-1)  # out : (B,C,num_sample_inout)
            xyz_feat_grid_list.append(xyz_feat_grid)

        multi_xyz_feat_grid = torch.cat(xyz_feat_grid_list, 1)

        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), multi_xyz_feat_grid, xyz, z_feat]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:, None].float() * self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            # print(preds, self.labels)
            error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)

        return error

    def forward(self, images, pcd, points, calibs, transforms=None, labels=None):
        # Get image feature
        self.filter(images)

        self.pcd_filter(pcd)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        # get the prediction
        res = self.get_preds()

        # get the error
        error = self.get_error()

        return res, error
