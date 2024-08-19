import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm2d(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class FeatExtractor(nn.Module):
    def __init__(self, base_chs, c_in=3, num_stage=3, use_dropout=False, use_bias=False):
        super(FeatExtractor, self).__init__()
        self.base_chs = base_chs
        self.num_stage = num_stage

        # cin->base_chs
        self.conv0 = nn.Sequential(
            nn.Conv2d(c_in, base_chs, kernel_size=7, padding=3, bias=use_bias),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(True)
        )
        self.conv1 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)
        self.conv2 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)

        # base_chs->base_chs*2
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_chs, base_chs*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*2),
            nn.ReLU(True)
        )
        self.conv4 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)
        self.conv5 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)

        # base_chs*2->base_chs*4
        self.conv6 = nn.Sequential(
            nn.Conv2d(base_chs * 2, base_chs * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*4),
            nn.ReLU(True)
        )
        self.conv7 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)
        self.conv8 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)
        
        # base_chs*2->base_chs*4
        self.conv9 = nn.Sequential(
            nn.Conv2d(base_chs * 4, base_chs * 8, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*8),
            nn.ReLU(True)
        )
        self.conv10 = ResnetBlock(base_chs*8, 'zero', use_dropout, use_bias)
        self.conv11 = ResnetBlock(base_chs*8, 'zero', use_dropout, use_bias)

        
        self.out4 = nn.Conv2d(base_chs*8, base_chs*8, 1, bias=False)
        self.out_channels = [base_chs*8]
                
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(base_chs*8, base_chs*4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*4),
            nn.ReLU(True)
        )
        self.deconv8 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)
        self.deconv9 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)
        
        self.out1 = nn.Conv2d(base_chs*4, base_chs*4, 1, bias=False)
        self.out_channels.append(base_chs*4)
        # self.out_channels = [base_chs*4]

        # base_chs*4->base_chs*2
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(base_chs*4, base_chs*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*2),
            nn.ReLU(True)
        )
        self.deconv2 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)
        self.deconv3 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)
        self.out2 = nn.Conv2d(base_chs*2, base_chs*2, 1, bias=False)
        self.out_channels.append(2 * base_chs)

        # base_chs*2->base_chs
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(base_chs*2, base_chs, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(True)
        )
        self.deconv5 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)
        self.deconv6 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)
        self.out3 = nn.Conv2d(base_chs, base_chs, 1, bias=False)
        self.out_channels.append(base_chs)
        
        # self.deconv7 = nn.Sequential(
        #     nn.ConvTranspose2d(base_chs*8, base_chs*4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
        #     nn.BatchNorm2d(base_chs),
        #     nn.ReLU(True)
        # )
        # self.deconv8 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)
        # self.deconv9 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)
        # self.out4 = nn.Conv2d(base_chs*4, base_chs*4, 1, bias=False)
        # self.out_channels.append(base_chs*4)


    def forward(self, x):
        out = self.conv0(x) # B 16 512 512
        out = self.conv1(out) # B 16 512 512
        out_d0 = self.conv2(out) # B 16 512 512
        out = self.conv3(out_d0) # B 32 256 256
        out = self.conv4(out) # B 32 256 256
        out_d1 = self.conv5(out) # B 32 256 256
        out = self.conv6(out_d1) # B 64 128 128
        out = self.conv7(out) # B 64 128 128
        out_d2 = self.conv8(out) # B 64 128 128
        out = self.conv9(out_d2) # B 128 64 64
        out = self.conv10(out) # B 128 64 64
        out = self.conv11(out) # B 128 64 64

        outputs = {}
        out4 = self.out4(out) # B 128 64 64
        outputs['stage1'] = out4

        out = self.deconv7(out) + out_d2 # B 64 128 128
        out = self.deconv8(out) # B 64 128 128
        out = self.deconv9(out) # B 64 128 128

        out1 = self.out1(out) # B 64 128 128
        outputs['stage2'] = out1

        out = self.deconv1(out) + out_d1 # B 32 256 256
        out = self.deconv2(out) # B 32 256 256
        out = self.deconv3(out) # B 32 256 256

        out2 = self.out2(out) # B 32 256 256
        outputs['stage3'] = out2
        
        out = self.deconv4(out) + out_d0 # B 16 512 512
        out = self.deconv5(out) # B 16 512 512
        out = self.deconv6(out) # B 16 512 512

        out3 = self.out3(out) # B 16 512 512
        outputs['stage4'] = out3
        
        return outputs, out



class NENet(nn.Module):
    def __init__(self, base_chs, fuse_type='max', c_in=3, use_dropout=False, use_bias=False):
        super(NENet, self).__init__()
        self.extractor = FeatExtractor(base_chs=base_chs, c_in=c_in, use_bias=use_bias)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.regressor = nn.Sequential(
            ResnetBlock(base_chs, 'zero', use_dropout, use_bias),
            ResnetBlock(base_chs, 'zero', use_dropout, use_bias),
            nn.Conv2d(base_chs, 3, kernel_size=7, padding=3)  # normal est
        )

        self.regressor2 = nn.Sequential(
            ResnetBlock(3*2, 'zero', use_dropout, use_bias),
            ResnetBlock(3*2, 'zero', use_dropout, use_bias),
            nn.Conv2d(3*2, 3, kernel_size=7, padding=3)  # normal est
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.view_dir = -torch.Tensor([0, 0, 1]).to('cuda')
        self.shininess = 32
        self.gauss_scale = 1
        self._mapping_size = 8
        self._B = torch.randn((16, self._mapping_size)).to('cuda') * self.gauss_scale

    def prepareInputs(self, x):
        imgs = x['imgs']
        b, v, l, c, h, w = imgs.shape
        single_view_imgs = torch.unbind(imgs, 1)  # (b, L, 3, h, w) along nviews dim
        single_view_dirs = torch.unbind(x['light_dirs'], 1)  # (b, l, 3) along nviews dim
        view_inputs = []
        for v in range(len(single_view_imgs)):  # for each view
            single_light_imgs = torch.unbind(single_view_imgs[v], 1)  # (b, 3, h, w) along l dim
            single_light_dirs = torch.unbind(single_view_dirs[v], 1)  # (b, 3) along l dim
            assert len(single_light_imgs) == len(single_light_dirs)  # l
            single_view_inputs = []
            for l in range(len(single_light_imgs)):  # for each light
                l_dir = single_light_dirs[l] if single_light_dirs[l].dim() == 4 else single_light_dirs[l].view(b, -1, 1, 1)  # (b, 3, 1, 1)
                img = single_light_imgs[l]  # (b, 3, h, w)
                img_light = torch.cat([img, l_dir.expand_as(img)], 1)  # (b,6,h,w)
                single_view_inputs.append(img_light)
            del single_light_imgs, single_light_dirs
            view_inputs.append(single_view_inputs)
        del single_view_imgs, single_view_dirs
        return view_inputs

    def forward(self, x):
        view_inputs = self.prepareInputs(x)
        view_feats_stage1 = []
        view_feats_stage2 = []
        view_feats_stage3 = []
        view_feats_stage4 = []
        view_normals = []
        for v in range(len(view_inputs)): # for views
            inputs = view_inputs[v]
            feats_stage1 = torch.Tensor()
            feats_stage2 = torch.Tensor()
            feats_stage3 = torch.Tensor()
            feats_stage4 = torch.Tensor()
            feats = torch.Tensor()
            for i in range(len(inputs)): # for lights
                outputs_dict, out = self.extractor(inputs[i])

            #     ################### 
                if i == 0:
                    feats_stage1 = outputs_dict['stage1'].unsqueeze(1)
                    feats_stage2 = outputs_dict['stage2'].unsqueeze(1)
                    feats_stage3 = outputs_dict['stage3'].unsqueeze(1)
                    feats_stage4 = outputs_dict['stage4'].unsqueeze(1)
                    feats = out.unsqueeze(1)
                else:
                    feats_stage1 = torch.cat([feats_stage1, outputs_dict['stage1'].unsqueeze(1)], 1)
                    feats_stage2 = torch.cat([feats_stage2, outputs_dict['stage2'].unsqueeze(1)], 1)
                    feats_stage3 = torch.cat([feats_stage3, outputs_dict['stage3'].unsqueeze(1)], 1)
                    feats_stage4 = torch.cat([feats_stage4, outputs_dict['stage4'].unsqueeze(1)], 1)
                    feats = torch.cat([feats, out.unsqueeze(1)], 1)

            if self.fuse_type == 'mean':
                feats = feats.mean(1)
            elif self.fuse_type == 'max':
                feats, _ = feats.max(1)

            '''
            ####### freq10
            B, C, H, W = feats.shape
            feats_fourier = feats.permute(0, 2, 3, 1).reshape(B * H * W, C)
            feats_fourier = feats_fourier @ (1 + self._B.to(feats.device))
            feats_fourier = feats_fourier.view(B, H, W, self._mapping_size).permute(0, 3, 1, 2) # B x C x H x W
            feats_fourier = 2 * torch.pi * feats_fourier
            feats_fourier = torch.cat([torch.sin(feats_fourier), torch.cos(feats_fourier)], dim=1)
            '''

            normal = self.regressor(feats)
            normal = torch.nn.functional.normalize(normal, dim=1)

            # # Prepare empty tensors to hold low and high frequency components
            # high_freq = torch.zeros_like(normal)

            # B, C, H, W = normal.shape
            # cutoff_frequency = 16

            # # Iterate over the color channels
            # for c in range(C):
            #     # 1. Perform 2D Fourier transform on each channel
            #     freq_channel = torch.fft.fft2(normal[:, c, :, :])

            #     # 2. Shift the zero-frequency component to the center of the spectrum
            #     freq_channel_shifted = torch.fft.fftshift(freq_channel)

            #     # 3. Create a mask for low frequencies
            #     rows, cols = H, W
            #     crow, ccol = rows // 2 , cols // 2  # center
            #     low_freq_mask = torch.zeros((H, W), dtype=torch.bool).to(normal.device)
            #     low_freq_mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = True

            #     # 4. Apply mask to separate low and high frequencies
            #     # low_freq_channel = freq_channel_shifted * low_freq_mask
            #     high_freq_channel = freq_channel_shifted * (~low_freq_mask)

            #     # 5. Inverse shift and Inverse Fourier transform
            #     # low_freq[:, c, :, :] = torch.fft.ifft2(torch.fft.ifftshift(low_freq_channel)).real
            #     high_freq[:, c, :, :] = torch.fft.ifft2(torch.fft.ifftshift(high_freq_channel)).real

            '''
            ####### freq8
            B, C, H, W = normal.shape
            normal_fourier = normal.permute(0, 2, 3, 1).reshape(B * H * W, C)
            normal_fourier = normal_fourier @ (1 + self._B.to(feats.device))
            normal_fourier = normal_fourier.view(B, H, W, self._mapping_size).permute(0, 3, 1, 2) # B x C x H x W
            normal_fourier = 2 * torch.pi * normal_fourier
            normal_fourier = torch.cat([torch.sin(normal_fourier), torch.cos(normal_fourier)], dim=1)
            normal = self.regressor2(normal_fourier)
            '''
            

            # normal = torch.nn.functional.normalize(high_freq, dim=1)
            view_normals.append(normal)
            del feats

            # normalT = normal.permute(0, 2, 3, 1)
            # _, H, W, _ = normalT.shape

            # light_dir = x['light_dirs'][:, v]
            # light_dir = torch.nn.functional.normalize(light_dir, dim=-1)

            # # Calculate dot products between normals and light directions: (H * W, 3) @ (L, 3).T -> (L, H * W)
            # dot_nl = torch.einsum('vhwz,vlz->vlhw', normalT, light_dir) # B x L x H x W

            # # Calculate the shadow maps by clamping negative values to 0
            # shadow_maps = torch.clamp(dot_nl, min=0.5, max=1)**0.2 # 1 x L x H x W

            # shadow_stage4 = shadow_maps
            # shadow_stage3 = torch.nn.functional.interpolate(shadow_maps, scale_factor=1/2, mode='bilinear', align_corners=False)
            # shadow_stage2 = torch.nn.functional.interpolate(shadow_maps, scale_factor=1/4, mode='bilinear', align_corners=False)
            # shadow_stage1 = torch.nn.functional.interpolate(shadow_maps, scale_factor=1/8, mode='bilinear', align_corners=False)

            # shadow_stage1 = shadow_stage1 / shadow_stage1.sum(1, keepdim=True)
            # shadow_stage2 = shadow_stage2 / shadow_stage2.sum(1, keepdim=True)
            # shadow_stage3 = shadow_stage3 / shadow_stage3.sum(1, keepdim=True)
            # shadow_stage4 = shadow_stage4 / shadow_stage4.sum(1, keepdim=True)

            # reflection_vectors = 2 * dot_nl.reshape(-1, len(inputs), H*W, 1) * normalT.reshape(-1, 1, H*W, 3) - light_dir.unsqueeze(2) # B x V x H*W x 3
            # reflection_vectors = torch.nn.functional.normalize(reflection_vectors, dim=-1)
            # reflection_vectors = reflection_vectors.reshape(-1, len(inputs), H, W, 3)

            # # Calculate specular intensity: (R • V)^shininess, result shape: (L, H * W)
            # specular_intensity = torch.einsum('nlhwz,nz->nlhw', reflection_vectors, self.view_dir[None].repeat(reflection_vectors.shape[0], 1))
            # specular_maps = torch.pow(torch.clamp(specular_intensity, min=0, max=0.5), self.shininess)
            # specular_maps = (1 - specular_maps) # B x L x H x W

            # specular_stage4 = specular_maps
            # specular_stage3 = torch.nn.functional.interpolate(specular_maps, scale_factor=1/2, mode='bilinear', align_corners=False)
            # specular_stage2 = torch.nn.functional.interpolate(specular_maps, scale_factor=1/4, mode='bilinear', align_corners=False)
            # specular_stage1 = torch.nn.functional.interpolate(specular_maps, scale_factor=1/8, mode='bilinear', align_corners=False)

            # specular_stage1 = specular_stage1 / specular_stage1.sum(1, keepdim=True)
            # specular_stage2 = specular_stage2 / specular_stage2.sum(1, keepdim=True)
            # specular_stage3 = specular_stage3 / specular_stage3.sum(1, keepdim=True)
            # specular_stage4 = specular_stage4 / specular_stage4.sum(1, keepdim=True)

            if self.fuse_type == 'mean':
                feats_stage1 = (feats_stage1).mean(1)
                feats_stage2 = (feats_stage2).mean(1)
                feats_stage3 = (feats_stage3).mean(1)
                feats_stage4 = (feats_stage4).mean(1)
                # feats_stage1 = (feats_stage1).mean(1)
                # feats_stage2 = (feats_stage2).mean(1)
                # feats_stage3 = (feats_stage3).mean(1)
                # feats_stage4 = (feats_stage4).mean(1)
            elif self.fuse_type == 'max':
                # feats_stage1 = (feats_stage1 * 1 * (specular_stage1).unsqueeze(2)).sum(1)
                # feats_stage2 = (feats_stage2 * 1 * (specular_stage2).unsqueeze(2)).sum(1)
                # feats_stage3 = (feats_stage3 * 1 * (specular_stage3).unsqueeze(2)).sum(1)
                # feats_stage4 = (feats_stage4 * 1 * (specular_stage4).unsqueeze(2)).sum(1)
                # feats_stage1, _ = (feats_stage1 * 1 * specular_stage1.unsqueeze(2)).max(1)
                # feats_stage2, _ = (feats_stage2 * 1 * specular_stage2.unsqueeze(2)).max(1)
                # feats_stage3, _ = (feats_stage3 * 1 * specular_stage3.unsqueeze(2)).max(1)
                # feats_stage4, _ = (feats_stage4 * 1 * specular_stage4.unsqueeze(2)).max(1)
                feats_stage1, _ = (feats_stage1).max(1)
                feats_stage2, _ = (feats_stage2).max(1)
                feats_stage3, _ = (feats_stage3).max(1)
                feats_stage4, _ = (feats_stage4).max(1)

            #     # ###################
            #     if i == 0:
            #         feats_stage1 = outputs_dict['stage1']
            #         feats_stage2 = outputs_dict['stage2']
            #         feats_stage3 = outputs_dict['stage3']
            #         feats_stage4 = outputs_dict['stage4']
            #         feats = out
            #     else:
            #         if self.fuse_type == 'mean':
            #             feats_stage1 = torch.stack([feats_stage1, outputs_dict['stage1']], 1).sum(1)
            #             feats_stage2 = torch.stack([feats_stage2, outputs_dict['stage2']], 1).sum(1)
            #             feats_stage3 = torch.stack([feats_stage3, outputs_dict['stage3']], 1).sum(1)
            #             feats_stage4 = torch.stack([feats_stage4, outputs_dict['stage4']], 1).sum(1)
            #             feats = torch.stack([feats, out], 1).sum(1)
            #         elif self.fuse_type == 'max':
            #             feats_stage1, _ = torch.stack([feats_stage1, outputs_dict['stage1']], 1).max(1)
            #             feats_stage2, _ = torch.stack([feats_stage2, outputs_dict['stage2']], 1).max(1)
            #             feats_stage3, _ = torch.stack([feats_stage3, outputs_dict['stage3']], 1).max(1)
            #             feats_stage4, _ = torch.stack([feats_stage4, outputs_dict['stage4']], 1).max(1)
            #             feats, _ = torch.stack([feats, out], 1).max(1)
            # if self.fuse_type == 'mean':
            #     feats_stage1 = feats_stage1 / len(inputs)
            #     feats_stage2 = feats_stage2 / len(inputs)
            #     feats_stage3 = feats_stage3 / len(inputs)
            #     feats_stage4 = feats_stage4 / len(inputs)
            #     feats = feats / len(inputs)
            view_feats_stage1.append(feats_stage1)
            view_feats_stage2.append(feats_stage2)
            view_feats_stage3.append(feats_stage3)
            view_feats_stage4.append(feats_stage4)

            # normal = self.regressor(feats)
            # normal = torch.nn.functional.normalize(normal, 2, 1)
            # view_normals.append(normal)
            # del feats
        view_normals = torch.stack(view_normals, 1)  # (b, v, 3, h, w)
        view_feats = {
            'stage1': view_feats_stage1,
            'stage2': view_feats_stage2,
            'stage3': view_feats_stage3,
            'stage4': view_feats_stage4
        }
        # view_normals: (b, v, 3, h, w)
        return view_feats, view_normals  # dict and tensor