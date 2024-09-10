from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
import scipy.io as sio
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from datasets.data_io import *



def normalize(a, axis=2):
    norm = np.sqrt((a * a).sum(axis, keepdims=True))
    a = a / (norm + 1e-10)
    return a


class MVSDataset(Dataset):
    def __init__(self, root_dir, split, nviews, ndepths=192, nlights=20, object='all', load_intrinsics=False,
                 use_sdps=False):
        super(MVSDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.nviews = nviews
        self.ndepth = ndepths
        self.nlights = nlights
        self.object = object
        self.load_intrinsics = load_intrinsics
        self.use_sdps = use_sdps

        self.img_wh = (512, 512)  # 'img_wh must both be multiples of 32!'
        assert split in ['train', 'val', 'test']

        if self.object =='all':
            self.scenes = ['bearPNG', 'buddhaPNG', 'cowPNG', 'pot2PNG', 'readingPNG']
        else:
            self.scenes = [self.object]

        self.lights = np.arange(1, 97)  # array

        # depth range in mm
        self.near_far = {
            'bearPNG': (1450, 1600),
            'buddhaPNG': (1450, 1600),
            'cowPNG': (1450, 1600),
            'pot2PNG': (1450, 1600),
            'readingPNG': (1450, 1600),
        }

        self.build_metas()
        self.read_params_file()

    def build_metas(self):
        self.metas = []
        for scene in self.scenes:
            with open('diligent_mv_pairs.txt') as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1:]]
                    # randomly sample lights
                    lights = sorted(random.sample(range(1, 97), self.nlights))
                    self.metas += [(scene, lights, ref_view, src_views)]

        print("dataset", self.split, "metas:", len(self.metas))

    def __len__(self):
        return len(self.metas)

    def read_params_file(self):
        self.params = {}
        for scene in self.scenes:
            self.params[scene] = sio.loadmat(os.path.join(self.root_dir, scene, 'Calib_Results.mat'))

    def read_img(self, filename, ints):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.dot(img, ints)
        # scale 0~65535 to 0~1
        # img = img.astype(np.float32) / 65535.
        # img = linear_to_srgb(img)
        img = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img))  # 0~1
        # img = reinhart(img)

        return img

    def read_depth(self, filename):
        depth, scale = read_pfm(filename)
        # depth = read_exr(filename, channel_names=['Y'])
        # depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[..., None]  # (h, w, 1)
        return depth

    def read_mask(self, filename):
        # depth = read_pfm(filename)
        # depth = read_exr(filename, channel_names=['Y'])
        depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[..., None]  # (h, w, 1)
        return depth
    
    def get_depth_z_axis(self, depth, fx, fy, H=512, W=512):
        if len(depth.shape) == 3:
            depth = depth.squeeze()
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i - W * .5) / fx, (j - H * .5) / fy, np.ones(i.shape, dtype=np.float32)], -1)
        dirs_length = np.linalg.norm(dirs, axis=-1)
        depth_z = depth / dirs_length
        return depth_z

    def get_depth(self, depth_filename, fx, fy):
        depth = self.read_depth(depth_filename)
        # depth = depth[:, 50:562]
        depth_stage3 = self.get_depth_z_axis(depth, fx, fy, 512, 512)
        depth_stage2 = cv2.resize(depth_stage3, None, fx=1.0 / 2, fy=1.0 / 2,)  # (256, 256)
        depth_stage1 = cv2.resize(depth_stage2, None, fx=1.0 / 2, fy=1.0 / 2,)  # (128, 128)
        depth = {
            'stage1': depth_stage1,
            'stage2': depth_stage2,
            'stage3': depth_stage3
        }
        return depth
    
    def get_depth_stage4(self, depth_filename, fx, fy):
        depth = self.read_depth(depth_filename)
        # depth = depth[:, 50:562]
        depth_stage4 = self.get_depth_z_axis(depth, fx, fy, 512, 512)
        depth_stage3 = cv2.resize(depth_stage4, None, fx=1.0 / 2, fy=1.0 / 2,)  # (256, 256)
        depth_stage2 = cv2.resize(depth_stage3, None, fx=1.0 / 2, fy=1.0 / 2,)  # (256, 256)
        depth_stage1 = cv2.resize(depth_stage2, None, fx=1.0 / 2, fy=1.0 / 2,)  # (128, 128)
        depth = {
            'stage1': depth_stage1,
            'stage2': depth_stage2,
            'stage3': depth_stage3,
            'stage4': depth_stage4
        }
        return depth

    def get_mask(self, filename, fx, fy):
        depth = self.read_mask(filename)
        depth = depth[:, 50:562]
        depth = self.get_depth_z_axis(depth, fx, fy, 512, 512)
        mask = depth > 0
        mask_stage3 = ndimage.binary_erosion(mask, structure=np.ones((7, 7))).astype(np.float32)
        mask_stage2 = cv2.resize(mask_stage3, None, fx=1.0 / 2, fy=1.0 / 2,)  # (256, 256)
        mask_stage1 = cv2.resize(mask_stage2, None, fx=1.0 / 2, fy=1.0 / 2,)  # (128, 128)
        return mask_stage1, mask_stage2, mask_stage3

    def get_mask_stage4(self, filename, fx, fy):
        depth = self.read_mask(filename)
        depth = depth[:, 50:562]
        depth = self.get_depth_z_axis(depth, fx, fy, 512, 512)
        mask = depth > 0
        mask_stage4 = ndimage.binary_erosion(mask, structure=np.ones((7, 7))).astype(np.float32)
        mask_stage3 = cv2.resize(mask_stage4, None, fx=1.0 / 2, fy=1.0 / 2,)  # (256, 256)
        mask_stage2 = cv2.resize(mask_stage3, None, fx=1.0 / 2, fy=1.0 / 2,)  # (128, 128)
        mask_stage1 = cv2.resize(mask_stage2, None, fx=1.0 / 2, fy=1.0 / 2,)  # (128, 128)
        return mask_stage1, mask_stage2, mask_stage3, mask_stage4

    def read_intrinsics(self, filename):
        with open(filename) as f:
            lines = f.readlines()
        intrinsics = np.fromstring(' '.join(lines), dtype=np.float32, sep=' ').reshape((3, 3))
        return intrinsics

    def read_extrinsics(self, filename):
        extrinsics = np.load(filename)
        return extrinsics

    def load_cam_params(self):
        xmls_path = os.path.join(self.root_dir, 'xmls', self.split)

        for scene in self.scenes:
            intrinsics = self.read_intrinsics(os.path.join(xmls_path, scene,
                                                           'intrinsics.txt'))
            extrinsics = self.read_extrinsics(os.path.join(xmls_path, scene,
                                                           'cam_to_worlds.npy'))
            if scene not in self.params:
                self.params[scene] = {}
            self.params[scene]['intrinsics'] = intrinsics
            self.params[scene]['c2w'] = extrinsics

    def load_light_dirs(self):
        ''' world coordinate system. '''
        xmls_path = os.path.join(self.root_dir, 'xmls', self.split)
        for scene in self.scenes:
            light_dirs = np.load(os.path.join(xmls_path, scene, 'cam_light_dirs.npy')).astype(np.float32)
            self.params[scene]['light_dirs'] = light_dirs  # (20, 10, 3)


    def make_scales(self, image, normalized=False, axis=2, mode=cv2.INTER_LINEAR):
        image_stage3 = image
        if normalized:
            image_stage3 = normalize(image_stage3, axis)
        image_stage2 = cv2.resize(image_stage3, None, fx=1.0/2, fy=1.0/2, interpolation=mode)
        if normalized:
            image_stage2 = normalize(image_stage2, axis)
        image_stage1 = cv2.resize(image_stage2, None, fx=1.0/2, fy=1.0/2, interpolation=mode)
        if normalized:
            image_stage1 = normalize(image_stage1, axis)

        return image_stage1, image_stage2, image_stage3
    
    def make_scales_stage4(self, image, normalized=False, axis=2, mode=cv2.INTER_LINEAR):
        image_stage4 = image
        if normalized:
            image_stage4 = normalize(image_stage4, axis)
        
        image_stage3 = cv2.resize(image_stage4, None, fx=1.0/2, fy=1.0/2, interpolation=mode)
        if normalized:
            image_stage3 = normalize(image_stage3, axis)
        
        image_stage2 = cv2.resize(image_stage3, None, fx=1.0/2, fy=1.0/2, interpolation=mode)
        if normalized:
            image_stage2 = normalize(image_stage2, axis)
            
        image_stage1 = cv2.resize(image_stage2, None, fx=1.0/2, fy=1.0/2, interpolation=mode)
        if normalized:
            image_stage1 = normalize(image_stage1, axis)

        return image_stage1, image_stage2, image_stage3, image_stage4
    

    def __getitem__(self, item):
        meta = self.metas[item]
        scene, lights, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        masks_stage1 = []
        masks_stage2 = []
        masks_stage3 = []
        masks_stage4 = []
        mask_erode = None

        ref_depth = None
        depth_values = None

        proj_matrices = [] ## for SnowMVS
        proj_mats_stage1 = []
        proj_mats_stage2 = []
        proj_mats_stage3 = []
        proj_mats_stage4 = []
        ref_proj_inv_stage4 = None
        ref_proj_inv_stage3 = None
        ref_proj_inv_stage2 = None
        ref_proj_inv_stage1 = None

        near = None
        far = None

        albedos_stage1 = []
        albedos_stage2 = []
        albedos_stage3 = []
        roughnesses = []
        
        normals_stage1 = []
        normals_stage2 = []
        normals_stage3 = []
        normals_stage4 = []
        
        light_dirs = []

        ###########################################
        shadows_stage1 = []
        shadows_stage2 = []
        shadows_stage3 = []
        shadows_stage4 = []

        speculars_stage1 = []
        speculars_stage2 = []
        speculars_stage3 = []
        speculars_stage4 = []
        ###########################################

        rotation_matrices = []
        
        ref_intrinsics = None

        mask_ms = None
        
        for i, vid in enumerate(view_ids):
            R = self.params[scene]['Rc_%d' % (vid + 1)].astype(np.float32).copy()  # w2c
            t = self.params[scene]['Tc_%d' % (vid + 1)].astype(np.float32).copy()  # w2c

            rotation_matrices.append(R)
            
            view_imgs = []
            view_dirs = []
            
            ## read light information : direction & intensity
            if not self.use_sdps:
                light_dir_filename = os.path.join(self.root_dir, scene, f'view_{vid + 1:02d}', 'light_directions.txt')
            else:
                light_dir_filename = os.path.join(self.root_dir, 'sdps_diligent_cal', scene, f'view_{vid + 1:02d}', 'light_dirs.txt')
            light_int_filename = os.path.join(self.root_dir, scene, f'view_{vid + 1:02d}', 'light_intensities.txt')
            l_dirs = np.genfromtxt(light_dir_filename).astype(np.float32)  # (96, 3)
            l_dirs = l_dirs * np.array([1, -1, -1]).astype(np.float32).reshape((1, 3))
            l_ints = np.genfromtxt(light_int_filename).astype(np.float32)  # (96, 3)

            ## read all images of the current view
            ints = [np.diag(1 / l_ints[light - 1]) for light in lights]
            for ind, light in enumerate(lights):  # for each light
                img_filename = os.path.join(self.root_dir, scene, f'view_{vid+1:02d}', f'{light:03d}.png')
                img = self.read_img(img_filename, ints[ind])
                img = img[:, 50:562, :]  # crop images from (512, 612) to (512, 512)
                view_imgs.append(img)
                view_dirs.append(l_dirs[light - 1])
            view_imgs = np.stack(view_imgs).transpose([0, 3, 1, 2])  # (L, 3, h, w)
            imgs.append(view_imgs) # stack all images from all views
            light_dirs.append(np.stack(view_dirs, 0))

            ##### extrinsics & intrinsics
            extrinsics = np.concatenate((R, t), axis=-1)
            intrinsics = self.params[scene]['KK'].astype(np.float32).copy()
            intrinsics[0][2] -= 50 # modify intrinsics because of image cropping
            proj_mat = np.zeros(shape=(2,4,4), dtype=np.float32)
            proj_mat[0, :3, :4] = extrinsics
            proj_mat[0, 3, 3] = 1
            proj_mat[1, :3, :3] = intrinsics
            
            proj_matrices.append(proj_mat)

            ## read mask and resize mask
            mask_filename = os.path.join(self.root_dir, scene, f'view_{vid+1:02d}', 'mask.png')
            masks = self.get_mask_stage4(mask_filename, intrinsics[0][0], intrinsics[1][1])
            masks_stage1.append(masks[0])  # (128, 128)
            masks_stage2.append(masks[1])
            masks_stage3.append(masks[2])
            masks_stage4.append(masks[3])

            ##### define depth range
            if i == 0:  # reference view
                near_far = self.near_far[scene]  # tuple (2,)
                t_vals = np.linspace(0., 1., num=self.ndepth, dtype=np.float32)  # (D)
                near, far = near_far
                depth_values = near * (1. - t_vals) + far * t_vals  # (D,)
                ref_intrinsics = intrinsics

                depth_filename = os.path.join(self.root_dir, scene, f'view_{vid+1:02d}', 'Depth_gt.pfm')
                # ref_depth = self.get_depth(depth_filename, intrinsics[0][0], intrinsics[1][1])
                ref_depth = self.get_depth_stage4(depth_filename, intrinsics[0][0], intrinsics[1][1])

            normal_filename = os.path.join(self.root_dir, scene, f'view_{vid+1:02d}', 'Normal_gt.mat')
            normal = sio.loadmat(normal_filename)['Normal_gt'].astype(np.float32)
            norm = np.sqrt((normal * normal).sum(2, keepdims=True))
            normal = normal / (norm + 1e-10)
            normal = normal[:, 50:562, :]
            normal = normal * np.array([1, -1, -1]).astype(np.float32).reshape((1, 1, 3))

            normals = self.make_scales_stage4(normal, normalized=True, axis=2)
            normals_stage1.append(normals[0])
            normals_stage2.append(normals[1])
            normals_stage3.append(normals[2])
            normals_stage4.append(normals[3])
    
            # ## read mask and resize mask
            # mask_filename = os.path.join(self.root_dir, scene, 'mask_depth', f'view_{vid+1:02d}.png')
            # mask = plt.imread(mask_filename)[:, 50:562]  # crop to (512, 512)
            # mask_stage3 = mask
            # mask_stage2 = cv2.resize(mask_stage3, None, fx=1.0 / 2, fy=1.0 / 2)
            # mask_stage1 = cv2.resize(mask_stage2, None, fx=1.0 / 2, fy=1.0 / 2)
            # masks_stage1.append(mask_stage1)  # (128, 128)
            # masks_stage2.append(mask_stage2)  # (h//2, w//2)
            # masks_stage3.append(mask_stage3)  # (h, w)




            # ##### proj_mats stage3
            # proj_mat_stage3 = extrinsics.copy()
            # proj_mat_stage3 = np.matmul(intrinsics, proj_mat_stage3)
            # proj_mat_stage3 = np.concatenate((proj_mat_stage3, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            # if i == 0:
            #     ref_proj_inv_stage3 = np.linalg.inv(proj_mat_stage3)
            #     proj_mats_stage3.append(np.eye(4).astype(np.float32))
            # else:
            #     proj_mats_stage3.append(proj_mat_stage3 @ ref_proj_inv_stage3)
                
            # ##### proj_mats stage2
            # intrinsics_stage2 = intrinsics.copy()
            # intrinsics_stage2[:2, :] = intrinsics_stage2[:2, :] / 2
            # proj_mat_stage2 = extrinsics.copy()
            # proj_mat_stage2 = np.matmul(intrinsics_stage2, proj_mat_stage2)
            # proj_mat_stage2 = np.concatenate((proj_mat_stage2, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            # if i == 0:
            #     ref_proj_inv_stage2 = np.linalg.inv(proj_mat_stage2)
            #     proj_mats_stage2.append(np.eye(4).astype(np.float32))
            # else:
            #     proj_mats_stage2.append(proj_mat_stage2 @ ref_proj_inv_stage2)
                
            # ##### proj_mats stage1
            # intrinsics_stage1 = intrinsics.copy()
            # intrinsics_stage1[:2, :] = intrinsics_stage1[:2, :] / 4
            # proj_mat_stage1 = extrinsics.copy()
            # proj_mat_stage1 = np.matmul(intrinsics_stage1, proj_mat_stage1)
            # proj_mat_stage1 = np.concatenate((proj_mat_stage1, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            # if i == 0:
            #     ref_proj_inv_stage1 = np.linalg.inv(proj_mat_stage1)
            #     proj_mats_stage1.append(np.eye(4).astype(np.float32))
            # else:
            #     proj_mats_stage1.append(proj_mat_stage1 @ ref_proj_inv_stage1)


            # normal_filename = os.path.join(self.root_dir, scene, f'view_{vid+1:02d}', 'Normal_gt.mat')
            # normal = sio.loadmat(normal_filename)['Normal_gt'].astype(np.float32)
            # norm = np.sqrt((normal * normal).sum(2, keepdims=True))
            # normal = normal / (norm + 1e-10)
            # normal = normal[:, 50:562, :]
            # normal = normal * np.array([1, -1, -1]).astype(np.float32).reshape((1, 1, 3))
            # normal_stage3 = normal
            # normal_stage2 = cv2.resize(normal_stage3, None, fx=1.0 / 2, fy=1.0 / 2)
            # normal_stage2 = normalize(normal_stage2, 2)
            # normal_stage1 = cv2.resize(normal_stage2, None, fx=1.0 / 2, fy=1.0 / 2)
            # normal_stage1 = normalize(normal_stage1, 2)
            # normals_stage1.append(normal_stage1)
            # normals_stage2.append(normal_stage2)
            # normals_stage3.append(normal_stage3)

        proj_matrices = np.stack(proj_matrices)

        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 8.0
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4.0
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        
        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats,
            "stage4": proj_matrices,
        }
        
        masks_stage1 = np.stack(masks_stage1)  # (V, H, W)
        masks_stage2 = np.stack(masks_stage2)
        masks_stage3 = np.stack(masks_stage3)
        masks_stage4 = np.stack(masks_stage4)
        masks = {
            'stage1': masks_stage1,
            'stage2': masks_stage2,
            'stage3': masks_stage3,
            'stage4': masks_stage4,
        }

        # stage1_pjmats = proj_matrices.copy()
        # stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4.0
        # stage2_pjmats = proj_matrices.copy()
        # stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        
        # proj_matrices_ms = {
        #     "stage1": stage1_pjmats,
        #     "stage2": stage2_pjmats,
        #     "stage3": proj_matrices,
        # }
        
        rotation_matrices = np.stack(rotation_matrices)
        
        imgs = np.stack(imgs)
        
        # proj_mats_stage1 = np.stack(proj_mats_stage1)
        # proj_mats_stage2 = np.stack(proj_mats_stage2)
        # proj_mats_stage3 = np.stack(proj_mats_stage3)
        # proj_mats = {
        #     'stage1': proj_mats_stage1,
        #     'stage2': proj_mats_stage2,
        #     'stage3': proj_mats_stage3,
        # }
        # masks_stage1 = np.stack(masks_stage1)  # (V, H, W)
        # masks_stage2 = np.stack(masks_stage2)
        # masks_stage3 = np.stack(masks_stage3)
        # masks = {
        #     'stage1': masks_stage1,
        #     'stage2': masks_stage2,
        #     'stage3': masks_stage3,
        # }
        normals_stage1 = np.stack(normals_stage1).transpose(0, 3, 1, 2)
        normals_stage2 = np.stack(normals_stage2).transpose(0, 3, 1, 2)
        normals_stage3 = np.stack(normals_stage3).transpose(0, 3, 1, 2)
        normals_stage4 = np.stack(normals_stage4).transpose(0, 3, 1, 2)
        normals = {
            'stage1': normals_stage1,
            'stage2': normals_stage2,
            'stage3': normals_stage3,
            'stage4': normals_stage4,
        }
        
        light_dirs = np.stack(light_dirs)  # (nviews, L , 3)

        sample = {}
        sample['imgs'] = imgs  # (nviews, L, 3, H, W)
        sample['proj_matrices'] = proj_matrices_ms  # dict, each (nviews, 4, 4)
        sample['depth'] = ref_depth  # ref_view, dict, (h//4, w//4), (h//2, w//2), (h, w)
        sample['depth_values'] = depth_values  # (ndepth, )
        sample['mask'] = masks  # dict, (nviews, h//4, w//4), (nviews, h//2, w//2), (nviews, h, w)
        sample['light_dirs'] = light_dirs  # (nview, L, 3)
        sample['filename'] = scene + '/{}/' + '{:0>2}'.format(view_ids[0]) + "{}"
        sample['R'] = rotation_matrices
        sample['normals'] = normals  # dict, (nviews, 3, h//4, w//4), (nviews, 3, h//2, w//2), (nviews, 3, h, w)
        if self.load_intrinsics:
            sample['intrinsics'] = ref_intrinsics

        return sample


if __name__ == '__main__':
    dataset = MVSDataset(
        root_dir="",
        split='test',
        nviews=4,
        nlights=2,
    )

    print(f"Dataset size: {len(dataset)}")
    item = dataset[0]

    # ----------------test homo_warping
    import pdb
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('..')
    from models.module import homo_warping
    import torch

    pdb.set_trace()

    imgs = torch.from_numpy(item['imgs']).unsqueeze(0)
    proj_matrices = torch.from_numpy(item['proj_matrices']['stage3']).unsqueeze(0)
    depth_values = torch.from_numpy(np.array(item['depth_values'])).unsqueeze(0)

    imgs = torch.unbind(imgs, 2)  # (B, V, 3, h, w)
    imgs = torch.unbind(imgs[0], 1)  # (B, 3, H, W)
    proj_matrices = torch.unbind(proj_matrices, 1)
    num_depth = depth_values.shape[1]
    ref_feature, src_features = imgs[0], imgs[1:]
    # ref_feature = imgs[0][:, :, ::4, ::4]
    # src_features = [imgs[i][:, :, ::4, ::4] for i in range(1, 3)]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
        # warpped features
        warped_volume = homo_warping(src_fea, src_proj, depth_values)

        for d in range(num_depth):
            print(depth_values[0, d])
            if d % 10 == 0:
                fig = plt.figure()
                # ax1 = fig.add_subplot(2, 1, 1)
                # ax1.imshow(ref_volume[0, :, d, :, :].permute(1, 2, 0).numpy())  # figure1
                # ax2 = fig.add_subplot(2, 1, 2)
                ax2 = fig.add_subplot(1, 1, 1)
                ax2.imshow(warped_volume[0, :, d, :, :].permute(1, 2, 0).numpy())  # figure2
                plt.savefig('')
                plt.close()