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
import torch.nn.functional as F
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

index_cache_global_dict = {
    'scale': None,
    'rot': None,
    'dc': None,
    'sh1': None,
    'sh2': None,
    'sh3': None
}

def render(
        viewpoint_camera,
        pc : GaussianModel,
        pipe,
        bg_color : torch.Tensor,
        vq_cfg: dict=None,
        scaling_modifier=1.0,
        override_color=None,
        activate_vq=False,
        activate_shprune=False,
        activate_gsprune=False,
        update_index=True
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    num_gs = means3D.shape[0]

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if activate_gsprune:
        # activated scale and opacity
        gs_prune_result_dict = pc.apply_gs_mask()
        scales = gs_prune_result_dict['scale']
        opacity = gs_prune_result_dict['opacity']
        gs_soft_mask = gs_prune_result_dict['gs_soft_mask']
        gs_hard_mask = gs_prune_result_dict['gs_hard_mask']
        gs_bitmask = gs_prune_result_dict['gs_bitmask']
        gs_mask_loss = torch.mean(gs_soft_mask)
        # calculate mask percentage
        gs_mask_percent = 1 - gs_hard_mask.sum() / gs_hard_mask.numel()
    else:
        gs_mask_loss = torch.tensor(0)
        gs_mask_percent = 0

    if activate_shprune:
        sh_prune_result_dict = pc.apply_sh_mask()
        features_rest = sh_prune_result_dict['features_rest']
        sh_soft_mask = sh_prune_result_dict['sh_soft_mask']
        sh_hard_mask = sh_prune_result_dict['sh_hard_mask']
        sh1_bitmask = sh_prune_result_dict['sh1_bitmask']
        sh2_bitmask = sh_prune_result_dict['sh2_bitmask']
        sh3_bitmask = sh_prune_result_dict['sh3_bitmask']
        if activate_gsprune:
            sh_hard_mask = sh_hard_mask[gs_bitmask]
            sh_soft_mask = sh_soft_mask[gs_bitmask]
            sh1_bitmask = torch.logical_and(sh1_bitmask, gs_bitmask)
            sh2_bitmask = torch.logical_and(sh2_bitmask, gs_bitmask)
            sh3_bitmask = torch.logical_and(sh3_bitmask, gs_bitmask)
        sh_mask_loss = torch.mean(sh_soft_mask)
        # calculate mask percentage
        sh_mask_percent = 1 - sh_hard_mask.sum() / sh_hard_mask.numel()
        shs[:, 1:, :] = features_rest
    else:
        sh_mask_loss = torch.tensor(0)
        sh_mask_percent = 0

    def ste(y_hat, y):
        return (y_hat - y).detach() + y

    rate_loss = [torch.tensor(0)]
    vq_loss = [torch.tensor(0)]
    bits_dict = {}
    index_cache_dict = {}
    vq_dim = 0
    
    if activate_vq:
        vq_inputs = pc.vector_dict
        vq_out = {}
        vq_inputs['sh1'] = features_rest[sh1_bitmask,:3,:].flatten(start_dim=1)
        vq_inputs['sh2'] = features_rest[sh2_bitmask,3:8,:].flatten(start_dim=1)
        vq_inputs['sh3'] = features_rest[sh3_bitmask,8:,:].flatten(start_dim=1)
        vq_inputs['scale'] = scales[gs_bitmask,:]
        vq_inputs['rot'] = vq_inputs['rot'][gs_bitmask,:]
        vq_inputs['dc'] = vq_inputs['dc'][gs_bitmask,:]
        if not update_index:
            # use cached indexes
            index_cache_dict['scale'] = index_cache_global_dict['scale'][gs_bitmask]
            index_cache_dict['rot'] = index_cache_global_dict['rot'][gs_bitmask]
            index_cache_dict['dc'] = index_cache_global_dict['dc'][gs_bitmask]
            index_cache_dict['sh1'] = index_cache_global_dict['sh1'][sh1_bitmask]
            index_cache_dict['sh2'] = index_cache_global_dict['sh2'][sh2_bitmask]
            index_cache_dict['sh3'] = index_cache_global_dict['sh3'][sh3_bitmask]
        
        for vq_key in vq_cfg['keys']:
            result = pc._quantizer[vq_key](vq_inputs[vq_key], index_cache=None if update_index else index_cache_dict[vq_key])
            vq_out[vq_key] = ste(y_hat=result['x_hat'], y=vq_inputs[vq_key])
            vq_out[vq_key] = pc.vq_post_process(vq_key, vq_out[vq_key])
            bits_dict[vq_key] = result['bits']
            rate_loss.append(bits_dict[vq_key] / vq_cfg['lmbda'][vq_key])
            index_cache_dict[vq_key] = result['x_index']
            vq_loss.append(torch.sum(torch.norm(result['x_hat'] - vq_inputs[vq_key], dim=-1)))
            vq_dim += vq_inputs[vq_key].shape[-1]
            
        scales[gs_bitmask,:] = vq_out['scale']
        rotations[gs_bitmask,:] = vq_out['rot']
        shs[gs_bitmask, 0:1, :] = vq_out['dc']
        shs[sh1_bitmask, 1:4, :] = vq_out['sh1']
        shs[sh2_bitmask, 4:9, :] = vq_out['sh2']
        shs[sh3_bitmask, 9:, :] = vq_out['sh3']
        
        if index_cache_global_dict['scale'] is None:
            for key in index_cache_global_dict.keys():
                index_cache_global_dict[key] = torch.zeros(num_gs, 1, device='cuda', dtype=torch.long)
        
        index_cache_global_dict['scale'][gs_bitmask] = index_cache_dict['scale']
        index_cache_global_dict['rot'][gs_bitmask] = index_cache_dict['rot']
        index_cache_global_dict['dc'][gs_bitmask] = index_cache_dict['dc']
        index_cache_global_dict['sh1'][sh1_bitmask] = index_cache_dict['sh1']
        index_cache_global_dict['sh2'][sh2_bitmask] = index_cache_dict['sh2']
        index_cache_global_dict['sh3'][sh3_bitmask] = index_cache_dict['sh3']

    if vq_dim != 0:
        rate_loss = sum(rate_loss) / num_gs / vq_dim
        vq_loss = sum(vq_loss) / num_gs / vq_dim
    else:
        rate_loss = sum(rate_loss) / num_gs
        vq_loss = sum(vq_loss) / num_gs

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "rate_loss": rate_loss,
        "vq_loss": vq_loss,
        "sh_mask_loss": sh_mask_loss,
        "sh_mask_percent": sh_mask_percent,
        "gs_mask_loss": gs_mask_loss,
        "gs_mask_percent": gs_mask_percent,
        "bits": bits_dict
    }