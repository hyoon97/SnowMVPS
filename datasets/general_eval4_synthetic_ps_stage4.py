from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import random
import scipy.ndimage as ndimage


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def normalize(a, axis=2):
    norm = np.sqrt((a * a).sum(axis, keepdims=True))
    a = a / (norm + 1e-10)
    return a

class MVSDataset(Dataset):
    def __init__(self, root_dir, split, nviews, ndepths=192, nlights=3, add_domains_to_load=None, load_intrinsics=False,
                 add_light_dir_augmentation=False, std=0.03):
        super(MVSDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.nviews = nviews
        self.ndepth = ndepths
        self.nlights = nlights
        self.add_domains_to_load = add_domains_to_load
        self.load_intrinsics = load_intrinsics
        self.add_light_dir_augmentation = add_light_dir_augmentation
        self.std = std

        self.img_wh = (512, 512)  # 'img_wh must both be multiples of 32!'
        assert split in ['train', 'val', 'test']

        self.scenes = sorted(os.listdir(os.path.join(self.root_dir,
                                                     'dataset',
                                                     self.split)))
        self.lights = np.arange(10)  # array

        self.build_metas()
        self.params = {}
        self.load_cam_params()
        if 'light_dir' in self.add_domains_to_load:
            self.load_light_dirs()

    def build_metas(self):
        self.metas = []
        for scene in self.scenes:
            with open(os.path.join(self.root_dir, 'synthetic_pairs.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    # print(f.readline().rstrip().split(', ')[1])
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1:]]
                    # randomly sample lights
                    lights = sorted(random.sample(range(0, 10), self.nlights))  # list (nlights,)
                    self.metas += [(scene, lights, ref_view, src_views)]

        print("dataset", self.split, "metas:", len(self.metas))

    def __len__(self):
        return len(self.metas)


    def read_img(self, filename):
        """Read exr image and normalize it."""
        # im = read_exr(filename)  # (512, 612, 3)
        im = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # needed if 3 channel image
        imd = (im - np.min(im)) / (np.max(im) - np.min(im))  # 0~1
        # imd = reinhart(im)
        return imd

    def read_depth(self, filename):
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
        depth = depth[:, 50:562]
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
        depth = depth[:, 50:562]
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
        depth = self.read_depth(filename)
        depth = depth[:, 50:562]
        depth = self.get_depth_z_axis(depth, fx, fy, 512, 512)
        mask = depth > 0
        mask_stage3 = ndimage.binary_erosion(mask, structure=np.ones((7, 7))).astype(np.float32)
        mask_stage2 = cv2.resize(mask_stage3, None, fx=1.0 / 2, fy=1.0 / 2,)  # (256, 256)
        mask_stage1 = cv2.resize(mask_stage2, None, fx=1.0 / 2, fy=1.0 / 2,)  # (128, 128)
        return mask_stage1, mask_stage2, mask_stage3

    def get_mask_stage4(self, filename, fx, fy):
        depth = self.read_depth(filename)
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
        
        for i, vid in enumerate(view_ids):  # for each view
            view_imgs = []
            for light in lights:  # for each light
                # NOTE that the id in image file names is from 0 to 19 (not 1~20)
                img_filename = os.path.join(self.root_dir, 'dataset', self.split,
                                            scene, f'image_{vid:03d}_l{light:02d}.exr')
                img = self.read_img(img_filename)
                # crop images from (512, 612) to (512, 512)
                view_imgs.append(img[:, 50:562, :])
            view_imgs = np.stack(view_imgs).transpose([0, 3, 1, 2])  # (L, 3, H, W)
            imgs.append(view_imgs)

            intrinsics = self.params[scene]['intrinsics'].copy()
            c2w = self.params[scene]['c2w'][vid].copy()  # (4, 4) # c2w
            # flip because of mitsuba coordinates
            # mitsuba: x -> left, y -> up, z -> towards
            c2w = c2w * np.array([-1, -1, 1, 1]).reshape((1, 4)).astype(np.float32)
            extrinsics = np.linalg.inv(c2w)
            
            rotation_matrices.append(extrinsics[:3, :3])
            
            # image cropping
            intrinsics[0][2] -= 50  # cx

            # # for SnowMVS only
            proj_mat = np.zeros(shape=(2,4,4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            
            #### proj_mats stage4
            intrinsics_stage4 = intrinsics.copy()
            proj_mat_stage4 = extrinsics[:3].copy()
            proj_mat_stage4 = np.matmul(intrinsics, proj_mat_stage4)
            proj_mat_stage4 = np.concatenate((proj_mat_stage4, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            if i == 0:
                ref_proj_inv_stage4 = np.linalg.inv(proj_mat_stage4)
                proj_mats_stage4.append(np.eye(4).astype(np.float32))
            else:
                proj_mats_stage4.append(proj_mat_stage4 @ ref_proj_inv_stage4)
                
            ##### proj_mats stage3
            intrinsics_stage3 = intrinsics.copy()
            intrinsics_stage3[:2, :] = intrinsics_stage3[:2, :] / 2
            proj_mat_stage3 = extrinsics[:3].copy()
            proj_mat_stage3 = np.matmul(intrinsics_stage3, proj_mat_stage3)
            proj_mat_stage3 = np.concatenate((proj_mat_stage3, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            if i == 0:
                ref_proj_inv_stage3 = np.linalg.inv(proj_mat_stage3)
                proj_mats_stage3.append(np.eye(4).astype(np.float32))
            else:
                proj_mats_stage3.append(proj_mat_stage3 @ ref_proj_inv_stage3)
                
            ##### proj_mats stage2
            intrinsics_stage2 = intrinsics.copy()
            # intrinsics_stage2[:2, :] = intrinsics_stage2[:2, :] / 2
            intrinsics_stage2[:2, :] = intrinsics_stage2[:2, :] / 4
            proj_mat_stage2 = extrinsics[:3].copy()
            proj_mat_stage2 = np.matmul(intrinsics_stage2, proj_mat_stage2)
            proj_mat_stage2 = np.concatenate((proj_mat_stage2, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            if i == 0:
                ref_proj_inv_stage2 = np.linalg.inv(proj_mat_stage2)
                proj_mats_stage2.append(np.eye(4).astype(np.float32))
            else:
                proj_mats_stage2.append(proj_mat_stage2 @ ref_proj_inv_stage2)
                
            ##### proj_mats stage1
            intrinsics_stage1 = intrinsics.copy()
            # intrinsics_stage1[:2, :] = intrinsics_stage1[:2, :] / 4
            intrinsics_stage1[:2, :] = intrinsics_stage1[:2, :] / 8
            proj_mat_stage1 = extrinsics[:3].copy()
            proj_mat_stage1 = np.matmul(intrinsics_stage1, proj_mat_stage1)
            proj_mat_stage1 = np.concatenate((proj_mat_stage1, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
            if i == 0:
                ref_proj_inv_stage1 = np.linalg.inv(proj_mat_stage2)
                proj_mats_stage1.append(np.eye(4).astype(np.float32))
            else:
                proj_mats_stage1.append(proj_mat_stage1 @ ref_proj_inv_stage1)

            mask_filename = os.path.join(self.root_dir, 'dataset', self.split,
                                          scene, f'depth_{vid:03d}.exr')

            # masks = self.get_mask(mask_filename, intrinsics[0][0], intrinsics[1][1])
            # masks_stage1.append(masks[0])  # (128, 128)
            # masks_stage2.append(masks[1])
            # masks_stage3.append(masks[2])
            
            masks = self.get_mask_stage4(mask_filename, intrinsics[0][0], intrinsics[1][1])
            masks_stage1.append(masks[0])  # (128, 128)
            masks_stage2.append(masks[1])
            masks_stage3.append(masks[2])
            masks_stage4.append(masks[3])


            if i == 0:  # reference view
                depth_filename = os.path.join(self.root_dir, 'dataset', self.split,
                                              scene, f'depth_{vid:03d}.exr')
                # ref_depth = self.get_depth(depth_filename, intrinsics[0][0], intrinsics[1][1])
                ref_depth = self.get_depth_stage4(depth_filename, intrinsics[0][0], intrinsics[1][1])

                near, far = 13, 17

                mask_erode = ndimage.binary_erosion(masks[2], structure=np.ones((3, 3)))

                t_vals = np.linspace(0., 1., num=self.ndepth, dtype=np.float32)
                depth_values = near * (1. - t_vals) + far * t_vals  # (D,)
                if self.load_intrinsics:
                    ref_intrinsics = intrinsics
                
                # mask_ref = self.get_mask(mask_filename, intrinsics[0][0], intrinsics[1][1])
                # mask_ms = {
                #     "stage1": masks[0],
                #     "stage2": masks[1],
                #     "stage3": masks[2],
                # }
                
                mask_ref = self.get_mask_stage4(mask_filename, intrinsics[0][0], intrinsics[1][1])
                mask_ms = {
                    "stage1": masks[0],
                    "stage2": masks[1],
                    "stage3": masks[2],
                    "stage4": masks[3],
                }

            if self.add_domains_to_load is not None:
                if 'albedo' in self.add_domains_to_load:
                    albedo_filename = os.path.join(self.root_dir, 'dataset', self.split, scene, 'albedo', f'albedo_{vid:03d}.exr')
                    # albedo = read_exr(albedo_filename)
                    albedo = cv2.imread(albedo_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    albedo = np.stack([albedo[..., 2], albedo[..., 1], albedo[..., 0]], -1)
                    albedo = albedo[:, 50:562, :]  # (h, w, 3)
                    albedos = self.make_scales(albedo)
                    albedos_stage1.append(albedos[0])
                    albedos_stage2.append(albedos[1])
                    albedos_stage3.append(albedos[2])
                if 'roughness' in self.add_domains_to_load:
                    roughness_filename = os.path.join(self.root_dir, 'dataset', self.split, scene, 'roughness', f'roughness_{vid:03d}.exr')
                    # roughness = read_exr(roughness_filename, channel_names=['Y'])
                    roughness = cv2.imread(roughness_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    roughness = roughness[..., None]
                    roughnesses.append(roughness[:, 50:562, :])  # (h, w, 1)
                    
                if 'normal' in self.add_domains_to_load:
                    normal_filename = os.path.join(self.root_dir, 'dataset', self.split, scene, f'normal_{vid:03d}.exr')
                    # normal = read_exr(normal_filename)
                    normal = cv2.imread(normal_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    normal = np.stack([normal[..., 2], normal[..., 1], normal[..., 0]], -1)
                    # # flip because of mitsuba coordinates
                    normal = normal * np.array([-1, -1, 1]).reshape((1, 1, 3)).astype(np.float32)
                    normal = normal[:, 50:562, :]
                    
                    # # # outputs stage1~3 scaled normal
                    # normals = self.make_scales(normal, normalized=True, axis=2)
                    # normals_stage1.append(normals[0])
                    # normals_stage2.append(normals[1])
                    # normals_stage3.append(normals[2])
                    
                    # outputs stage1~4 scaled normal
                    normals = self.make_scales_stage4(normal, normalized=True, axis=2)
                    normals_stage1.append(normals[0])
                    normals_stage2.append(normals[1])
                    normals_stage3.append(normals[2])
                    normals_stage4.append(normals[3])
                    
                if 'light_dir' in self.add_domains_to_load:
                    view_lights = []
                    for light in lights:
                        # make light dir in camera coordinate system
                        light_dir = self.params[scene]['light_dirs'][vid][light]
                        light_dir = (-1) * np.matmul(extrinsics[:3, :3], light_dir)
                        if self.add_light_dir_augmentation:
                            cov = self.std ** 2
                            noise = np.random.multivariate_normal([0,0,0],
                                                                  np.diag([cov, cov, cov]))
                            light_dir += noise
                            light_dir /= np.linalg.norm(light_dir)
                        view_lights.append(light_dir)
                    view_lights = np.stack(view_lights)  # (L, 3)
                    light_dirs.append(view_lights)

                ###########################################
                if len(light_dirs) > 0 and len(normals_stage1) > 0:
                    H, W, _ = normal.shape
                    normal_reshaped = normal.reshape(-1, 3)

                    light_dir = light_dir / np.linalg.norm(light_dir)
                    light_dir = light_dir[..., None]

                    # Calculate dot product between each normal and each light direction
                    shadow = np.dot(normal_reshaped, light_dir)

                    # Reshape shadow maps back to H x W x 3
                    shadow = shadow.reshape(H, W)
                    
                    # Set negative values to zero (indicating shadow regions)
                    shadow[shadow < 0] = 0
                    shadows = self.make_scales_stage4(shadow, normalized=False, axis=2)
                    shadows_stage1.append(shadows[0][..., None])
                    shadows_stage2.append(shadows[1][..., None])
                    shadows_stage3.append(shadows[2][..., None])
                    shadows_stage4.append(shadows[3][..., None])

                    view_dir = -np.array([0, 0, 1])

                    # Calculate reflection vector: R = 2(N • L)N - L
                    dot_nl = np.dot(normal_reshaped, light_dir)
                    reflection_vector = 2 * dot_nl * normal_reshaped - light_dir.T
                    
                    # Normalize reflection vector
                    reflection_vector = reflection_vector / np.linalg.norm(reflection_vector, axis=1, keepdims=True)
                    
                    shininess = 32

                    # Calculate specular intensity: (R • V)^shininess
                    specular_intensity = np.dot(reflection_vector, view_dir)
                    specular_intensity = np.maximum(specular_intensity, 0)  # Clamp negative values to 0
                    specular = np.power(specular_intensity, shininess)
                    specular = specular.reshape(H, W)

                    speculars = self.make_scales_stage4(specular, normalized=False, axis=2)
                    speculars_stage1.append(speculars[0][..., None])
                    speculars_stage2.append(speculars[1][..., None])
                    speculars_stage3.append(speculars[2][..., None])
                    speculars_stage4.append(speculars[3][..., None])
                    ###########################################

        imgs = np.stack(imgs)  # (V, L, 3, H, W)
        
        # proj_matrices = np.stack(proj_matrices)
        # stage1_pjmats = proj_matrices.copy()
        # stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4.0
        # stage2_pjmats = proj_matrices.copy()
        # stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        # stage3_pjmats = proj_matrices.copy()
        
        # proj_matrices_ms = {
        #     "stage1": stage1_pjmats,
        #     "stage2": stage2_pjmats,
        #     "stage3": stage3_pjmats,
        # }
        
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
        
        proj_mats_stage1 = np.stack(proj_mats_stage1)
        proj_mats_stage2 = np.stack(proj_mats_stage2)
        proj_mats_stage3 = np.stack(proj_mats_stage3)
        proj_mats_stage4 = np.stack(proj_mats_stage4)
        proj_mats = {
            'stage1': proj_mats_stage1,
            'stage2': proj_mats_stage2,
            'stage3': proj_mats_stage3,
            'stage4': proj_mats_stage4,
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

        rotation_matrices = np.stack(rotation_matrices)
        
        sample = {}
        sample['imgs'] = imgs  # (nviews, L, 3, H, W)
        # sample['proj_matrices'] = proj_mats  # dict, each (nviews, 4, 4)
        sample['proj_matrices'] = proj_matrices_ms  # dict, each (nviews, 4, 4)
        sample['depth'] = ref_depth  # ref_view, dict, (h//4, w//4), (h//2, w//2), (h, w)
        sample['depth_values'] = depth_values  # (ndepth, )
        sample['mask'] = masks  # dict, (nviews, h//4, w//4), (nviews, h//2, w//2), (nviews, h, w)
        # sample['mask'] = mask_ms
        sample['mask_erode'] = mask_erode  # ref view, eroded, (h, w)
        sample['near'] = near
        sample['far'] = far
        sample['R'] = rotation_matrices

        if 'albedo' in self.add_domains_to_load:
            albedos_stage1 = np.stack(albedos_stage1).transpose(0, 3, 1, 2)
            albedos_stage2 = np.stack(albedos_stage2).transpose(0, 3, 1, 2)
            albedos_stage3 = np.stack(albedos_stage3).transpose(0, 3, 1, 2)
            albedos = {
                'stage1': albedos_stage1,
                'stage2': albedos_stage2,
                'stage3': albedos_stage3,
            }
            sample['albedos'] = albedos  # dict, (nviews, h//4, w//4), (nviews, h//2, w//2), (nviews, h, w)
        if 'roughness' in self.add_domains_to_load:
            roughnesses = np.stack(roughnesses).transpose(0, 3, 1, 2)  # (nviews, 1, h, w)
            sample['roughnesses'] = roughnesses
        if 'normal' in self.add_domains_to_load:
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
            sample['normals'] = normals  # dict, (nviews, 3, h//4, w//4), (nviews, 3, h//2, w//2), (nviews, 3, h, w)
        if 'light_dir' in self.add_domains_to_load:
            sample['light_dirs'] = np.stack(light_dirs)  # (nview, L, 3)

        else:
            sample['light_dirs'] = np.zeros((self.nviews, self.nlights, 3), dtype=np.float32)

        if self.load_intrinsics:
            sample['intrinsics'] = ref_intrinsics
            
        sample['filename'] = scene + '/{}/' + '{:0>2}'.format(view_ids[0]) + "{}"
        
        # ###########################################
        # if len(light_dirs) > 0 and len(shadows_stage1) > 0:
            
        #     shadows_stage1 = np.stack(shadows_stage1).transpose(0, 3, 1, 2)
        #     shadows_stage2 = np.stack(shadows_stage2).transpose(0, 3, 1, 2)
        #     shadows_stage3 = np.stack(shadows_stage3).transpose(0, 3, 1, 2)
        #     # normals_stage4 = np.stack(normals_stage4).transpose(0, 3, 1, 2)
        #     shadows = {
        #         'stage1': shadows_stage1,
        #         'stage2': shadows_stage2,
        #         'stage3': shadows_stage3,
        #         'stage4': shadows_stage4,
        #     }
        #     sample['shadows'] = shadows  # dict, (nviews, 3, h//4, w//4), (nviews, 3, h//2, w//2), (nviews, 3, h, w)
    
        #     speculars_stage1 = np.stack(speculars_stage1).transpose(0, 3, 1, 2)
        #     speculars_stage2 = np.stack(speculars_stage2).transpose(0, 3, 1, 2)
        #     speculars_stage3 = np.stack(speculars_stage3).transpose(0, 3, 1, 2)
        #     # normals_stage4 = np.stack(normals_stage4).transpose(0, 3, 1, 2)
        #     speculars = {
        #         'stage1': speculars_stage1,
        #         'stage2': speculars_stage2,
        #         'stage3': speculars_stage3,
        #         'stage4': speculars_stage4,
        #     }
        #     sample['speculars'] = speculars  # dict, (nviews, 3, h//4, w//4), (nviews, 3, h//2, w//2), (nviews, 3, h, w)
        # ###########################################

        return sample