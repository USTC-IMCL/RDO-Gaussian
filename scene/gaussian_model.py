#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from vq.ecvq import ECVQ
from utils.encode_utils import *

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, vq_cfg:dict=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.vq_cfg = vq_cfg
        self.gs_mask_thres = 0.1
        self.sh_mask_thres = 0.1

        self._quantizer = nn.ModuleDict()

    @property
    def vector_dict(self):
        return {
            'scale': self.get_scaling.flatten(start_dim=1),
            'rot': self._rotation.flatten(start_dim=1),
            'dc': self._features_dc.flatten(start_dim=1),
            'sh1': self._features_rest.tensor_split((3,8), dim=1)[0].flatten(start_dim=1),
            'sh2': self._features_rest.tensor_split((3,8), dim=1)[1].flatten(start_dim=1),
            'sh3': self._features_rest.tensor_split((3,8), dim=1)[2].flatten(start_dim=1)
        }
    
    def vq_post_process(self, key, value):
        if key == 'scale':
            return value
        if key == 'rot':
            return self.rotation_activation(value)
        if key == 'dc':
            return torch.unsqueeze(value, 1)
        if key == 'sh1':
            return torch.reshape(value, (-1, 3, 3))
        if key == 'sh2':
            return torch.reshape(value, (-1, 5, 3))
        if key == 'sh3':
            return torch.reshape(value, (-1, 7, 3))

    def initialize_quantizer(self):
        vq_cfg = self.vq_cfg
        x_dim = {
            'scale': self._scaling.flatten(start_dim=1).shape[-1],
            'rot': self._rotation.flatten(start_dim=1).shape[-1],
            'dc': self._features_dc.flatten(start_dim=1).shape[-1],
            'sh1': self._features_rest.tensor_split((3,8), dim=1)[0].flatten(start_dim=1).shape[-1],
            'sh2': self._features_rest.tensor_split((3,8), dim=1)[1].flatten(start_dim=1).shape[-1],
            'sh3': self._features_rest.tensor_split((3,8), dim=1)[2].flatten(start_dim=1).shape[-1]
        }
        for key in vq_cfg['keys']:
            self._quantizer[key] = ECVQ(
                x_dim=x_dim[key], cb_dim=x_dim[key], cb_size=vq_cfg['cb_size'][key],
                lmbda=vq_cfg['lmbda'][key], patch_size=vq_cfg['patch_size'],
                rate_constrain=False,
            ).cuda()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.sh_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 1, device="cuda").requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.sh_mask = nn.Parameter(torch.zeros(features.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask = nn.Parameter(torch.zeros(features.shape[0], 1, device="cuda").requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        l = [
            {'params': [self._quantizer[key].codebook for key in self._quantizer.keys()], 'lr': self.vq_cfg['cb_lr']},
            {'params': [self._quantizer[key].logits for key in self._quantizer.keys()], 'lr': self.vq_cfg['logits_lr']}
        ]
        self.optimizer_vq = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler_vq = MultiStepLR(self.optimizer_vq, milestones=[12000], gamma=0.1)
        l = [
            {'params': [self.sh_mask], 'lr': training_args.sh_mask_lr, "name": "sh_mask"}
        ]
        self.optimizer_sh_mask = torch.optim.Adam(l, lr=training_args.sh_mask_lr, eps=1e-15)
        l = [
            {'params': [self.gs_mask], 'lr': training_args.gs_mask_lr, "name": "gs_mask"}
        ]
        self.optimizer_gs_mask = torch.optim.Adam(l, lr=training_args.gs_mask_lr, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def apply_and_save_vq(self, vq_path):
        mkdir_p(vq_path)
        
        # Remove pruned Gaussians
        gs_bitmask = self.apply_gs_mask()['gs_bitmask']
        self.prune_points(~gs_bitmask)
        sh_bitmask = {}
        sh_prune_result_dict = self.apply_sh_mask()
        for sh_key in ['sh1', 'sh2', 'sh3']:
            sh_bitmask[sh_key] = sh_prune_result_dict[f'{sh_key}_bitmask'].detach().contiguous().cpu().numpy()
        
        # Apply VQ
        num_gaussians = self._xyz.shape[0]
        vq_out = {}
        vq_indexes = {}
        vq_codebooks = {}
        vq_logits = {}
        remove_negatives(self._quantizer['scale'].codebook)
        for vq_key in self.vq_cfg['keys']:
            if vq_key in ['sh1', 'sh2', 'sh3']:
                vq_result = self._quantizer[vq_key](self.vector_dict[vq_key][sh_bitmask[vq_key]])
            else:
                vq_result = self._quantizer[vq_key](self.vector_dict[vq_key])
            vq_xhat = self.vq_post_process(vq_key, vq_result["x_hat"]) if vq_key != 'rot' else vq_result["x_hat"]
            vq_out[vq_key] = vq_xhat
            vq_indexes[vq_key] = vq_result['x_index'].detach().contiguous().cpu().numpy()
            vq_codebooks[vq_key] = self._quantizer[vq_key].codebook.detach().contiguous().cpu().numpy()
            vq_logits[vq_key] = self._quantizer[vq_key].logits.detach().contiguous().cpu().numpy()
            
        features_rest = torch.zeros_like(self._features_rest.data)
        features_rest[sh_bitmask['sh1'],:3,:] = vq_out['sh1']
        features_rest[sh_bitmask['sh2'],3:8,:] = vq_out['sh2']
        features_rest[sh_bitmask['sh3'],8:,:] = vq_out['sh3']
        self._features_rest.data = features_rest
        self._features_dc.data = vq_out['dc']
        self._scaling.data = self.scaling_inverse_activation(vq_out['scale'])
        self._rotation.data = vq_out['rot']
        
        # Rearrange Gaussians by SH masks
        sort_idx, boundaries = shmask_sort(sh_bitmask)
        sh = -np.ones((num_gaussians, 3), dtype=int)
        sh[sh_bitmask['sh1'], 0] = vq_indexes['sh1'].squeeze()
        sh[sh_bitmask['sh2'], 1] = vq_indexes['sh2'].squeeze()
        sh[sh_bitmask['sh3'], 2] = vq_indexes['sh3'].squeeze()
        
        sh = sh[sort_idx]
            
        vq_indexes['sh1'] = sh[boundaries[3]:,0:1]
        vq_indexes['sh2'] = np.concatenate([sh[boundaries[1]:boundaries[3], 1:2],
                                            sh[boundaries[5]:, 1:2]])
        vq_indexes['sh3'] = np.concatenate([sh[boundaries[0]:boundaries[1], 2:],
                                            sh[boundaries[2]:boundaries[3], 2:],
                                            sh[boundaries[4]:boundaries[5], 2:],
                                            sh[boundaries[6]:, 2:]])
        for key in ['scale', 'rot', 'dc']:
            vq_indexes[key] = vq_indexes[key][sort_idx]
        vq_codebooks, vq_logits, vq_indexes = shrink_codebook(vq_codebooks, vq_logits, vq_indexes)
        
        # Non-VQ attributes
        xyz = self._xyz.detach().cpu().numpy().astype(np.float16)[sort_idx]
        opacities = self._opacity.detach().cpu().numpy()[sort_idx]
        quantized_opacities, opacities, opacity_log_prob, step_size, min_opacity = opacity_quant(opacities)
        vq_logits['opa'] = opacity_log_prob

        # Arithmetic coding
        index_strings = []
        index_lengths = []
            
        for vq_key in vq_indexes.keys():
            bits_str = entropy_coding(torch.Tensor(vq_indexes[vq_key]), torch.Tensor(vq_logits[vq_key]))
            index_strings.append(bits_str)
            index_lengths.append(len(bits_str))
            
        bits_str = entropy_coding(torch.Tensor(quantized_opacities), torch.Tensor(opacity_log_prob))
        index_strings.append(bits_str)
        index_lengths.append(len(bits_str))
        
        index_bitstream = pack_strings(index_strings)
        boundaries_bitstream = pack_uints(boundaries.tolist())
        opacity_header_bitstream = pack_floats([step_size, min_opacity])
                
        with open(os.path.join(vq_path, 'index_bitstream.bin'), "wb") as f:
            f.write(index_bitstream)
        with open(os.path.join(vq_path, 'header.bin'), "wb") as f:
            f.write(boundaries_bitstream)
            f.write(opacity_header_bitstream)
                        
        np.savez(os.path.join(vq_path, 'codebook.npz'), **vq_codebooks)
        np.savez(os.path.join(vq_path, 'logits.npz'), **vq_logits)
        np.savez(os.path.join(vq_path, 'position.npz'), position=xyz)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
    def decode(self, bitstream_path):
        xyz = np.load(os.path.join(bitstream_path, 'position.npz'))['position']
        vq_codebooks = np.load(os.path.join(bitstream_path, 'codebook.npz'))
        logits = np.load(os.path.join(bitstream_path, 'logits.npz'))
        boundaries, step_size, min_opacity = decode_header(os.path.join(bitstream_path, 'header.bin'))
        shape_dict = get_index_shapes(boundaries)
        index_path = os.path.join(bitstream_path, 'index_bitstream.bin')
        indexes = decode_indexes(index_path, logits, shape_dict)
        sh_bitmask = get_sh_bitmask(boundaries)
        opacities = opacity_dequant(indexes.pop('opa'), min_opacity, step_size).to('cuda')
        
        vq_attributes = {}
        
        for vq_key in vq_codebooks.files:
            codebook = vq_codebooks[vq_key]
            index = indexes[vq_key].squeeze(1)
            attr = codebook[0, index, :]
            attr = torch.tensor(attr, dtype=torch.float, device="cuda")
            vq_attributes[vq_key] = self.vq_post_process(vq_key, attr) if vq_key not in ['rot', 'scale'] else attr

        num_gaussians = boundaries[-1]
        features_dc = vq_attributes['dc']
        features_extra = torch.zeros((num_gaussians, 15, 3), dtype=torch.float, device="cuda")
        features_extra[sh_bitmask['sh1'], :3, :] = vq_attributes['sh1']
        features_extra[sh_bitmask['sh2'], 3:8, :] = vq_attributes['sh2']
        features_extra[sh_bitmask['sh3'], 8:, :] = vq_attributes['sh3']
        scales = self.scaling_inverse_activation(vq_attributes['scale'])
        rots = vq_attributes['rot']
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_extra)
        self._opacity = nn.Parameter(opacities)
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        
        self.active_sh_degree = self.max_sh_degree
        
    
    def load_vq(self, vq_path):
        xyz = np.load(os.path.join(vq_path, 'position.npz'))['position']
        opacities = np.load(os.path.join(vq_path, 'opacity.npz'))['opacity']
        num_gaussians = xyz.shape[0]
        vq_codebooks = np.load(os.path.join(vq_path, 'codebook.npz'))
        vq_indexes = np.load(os.path.join(vq_path, 'index.npz'))
        if os.path.exists(os.path.join(vq_path, 'sh_bitmask.npz')):
            sh_bitmask = np.load(os.path.join(vq_path, 'sh_bitmask.npz'))
            sh_bitmask = {sh_key: sh_bitmask[sh_key] for sh_key in ['sh1', 'sh2', 'sh3']}
        elif os.path.exists(os.path.join(vq_path, 'header.npz')):
            boundaries = np.load(os.path.join(vq_path, 'header.npz'))['boundaries']
            shmask_sorted = np.zeros(num_gaussians, dtype=int)
            for i in range(7, -1, -1):
                shmask_sorted[:boundaries[i]] = i
            
            sh_bitmask = {sh_key: np.zeros(num_gaussians, dtype=bool) for sh_key in ['sh1', 'sh2', 'sh3']}

            sh_bitmask['sh1'] |= (shmask_sorted & (1 << 2)).astype(bool)
            sh_bitmask['sh2'] |= (shmask_sorted & (1 << 1)).astype(bool)
            sh_bitmask['sh3'] |= (shmask_sorted & (1 << 0)).astype(bool)
        else:
            sh_bitmask = {sh_key: np.ones((num_gaussians, ), dtype=np.bool_) for sh_key in ['sh1', 'sh2', 'sh3']}
        vq_attributes = {}
        
        for vq_key in vq_codebooks.files:
            codebook = vq_codebooks[vq_key]
            index = vq_indexes[vq_key].squeeze(1)
            attr = codebook[0, index, :]
            attr = torch.tensor(attr, dtype=torch.float, device="cuda")
            vq_attributes[vq_key] = self.vq_post_process(vq_key, attr) if vq_key not in ['rot', 'scale'] else attr
            
        features_dc = vq_attributes['dc']
        features_extra = torch.zeros((num_gaussians, 15, 3), dtype=torch.float, device="cuda")
        features_extra[sh_bitmask['sh1'], :3, :] = vq_attributes['sh1']
        features_extra[sh_bitmask['sh2'], 3:8, :] = vq_attributes['sh2']
        features_extra[sh_bitmask['sh3'], 8:, :] = vq_attributes['sh3']
        scales = self.scaling_inverse_activation(vq_attributes['scale'])
        rots = vq_attributes['rot']
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_extra)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        
        self.active_sh_degree = self.max_sh_degree

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.sh_mask = nn.Parameter(torch.zeros(xyz.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask = nn.Parameter(torch.zeros(xyz.shape[0], 1, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                
        for group in self.optimizer_gs_mask.param_groups:
            stored_state = self.optimizer_gs_mask.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer_gs_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer_gs_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                
        for group in self.optimizer_sh_mask.param_groups:
            stored_state = self.optimizer_sh_mask.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer_sh_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer_sh_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.gs_mask = optimizable_tensors["gs_mask"]
        self.sh_mask = optimizable_tensors["sh_mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        for group in self.optimizer_gs_mask.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer_gs_mask.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer_gs_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer_gs_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        for group in self.optimizer_sh_mask.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer_sh_mask.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer_sh_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer_sh_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_gs_mask, new_sh_mask):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "gs_mask": new_gs_mask,
        "sh_mask": new_sh_mask
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.gs_mask = optimizable_tensors["gs_mask"]
        self.sh_mask = optimizable_tensors["sh_mask"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_gs_mask = self.gs_mask[selected_pts_mask].repeat(N,1)
        new_sh_mask = self.sh_mask[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_gs_mask, new_sh_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_gs_mask = self.gs_mask[selected_pts_mask]
        new_sh_mask = self.sh_mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_gs_mask, new_sh_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def apply_sh_mask(self):
        sh_soft_mask = torch.sigmoid(torch.cat([self.sh_mask[:, 0:1].unsqueeze(2).expand(-1, 3, 3),
            self.sh_mask[:, 1:2].unsqueeze(2).expand(-1, 5, 3),
            self.sh_mask[:, 2:3].unsqueeze(2).expand(-1, 7, 3)], dim=1))
        sh_hard_mask = ((sh_soft_mask > self.sh_mask_thres).float() - sh_soft_mask).detach() + sh_soft_mask
        features_rest = torch.mul(self._features_rest, sh_hard_mask)
        
        return {
            'features_rest': features_rest,
            'sh_soft_mask': sh_soft_mask,
            'sh_hard_mask': sh_hard_mask,
            'sh1_bitmask': torch.sigmoid(self.sh_mask[:, 0]) > self.sh_mask_thres,
            'sh2_bitmask': torch.sigmoid(self.sh_mask[:, 1]) > self.sh_mask_thres,
            'sh3_bitmask': torch.sigmoid(self.sh_mask[:, 2]) > self.sh_mask_thres,
        }
        
    def apply_gs_mask(self):
        gs_soft_mask = torch.sigmoid(self.gs_mask)
        gs_hard_mask = ((gs_soft_mask > self.gs_mask_thres).float() - gs_soft_mask).detach() + gs_soft_mask
        
        scales = torch.mul(self.get_scaling, gs_hard_mask)
        opacity = torch.mul(self.get_opacity, gs_hard_mask)
        return {
            'scale': scales, 
            'opacity': opacity,
            'gs_soft_mask': gs_soft_mask,
            'gs_hard_mask': gs_hard_mask,
            'gs_bitmask': gs_soft_mask.flatten() > self.gs_mask_thres
        }