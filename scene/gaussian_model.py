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
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.vq_utils import get_weights_and_indices, softmax_to_topk_soft_code

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


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._language_feature_logits = None
        self._language_feature_codebooks = None
        self._language_feature_weights = None
        self._language_feature_indices = None
        # Optional evaluation-only lineage.  It is deliberately excluded from
        # capture() and deployment artifacts; train_joint writes a separately
        # hashed sidecar for controlled source-ID runs.
        self._source_ids = None
        self._source_id_origin_count = None
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self, include_feature=False):
        if include_feature:
            assert self._language_feature_logits is not None, "language feature logits is None"
            assert self._language_feature_codebooks is not None, "language feature codebooks is None"
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature_logits,
                self._language_feature_codebooks,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
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
    
    def restore(self, model_args, training_args, mode='train'):
        if len(model_args) == 14: # for language feature
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._language_feature_logits,
            self._language_feature_codebooks,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif len(model_args) == 12:
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
            if not training_args.include_feature:
                self.optimizer.load_state_dict(opt_dict)
        
        if mode == 'train':
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
        

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    def enable_source_id_tracking(self, source_ids=None, origin_point_count=None):
        """Attach a strictly ordered source-row ID to every current Gaussian."""
        point_count = int(self._xyz.shape[0])
        if source_ids is None:
            source_ids = torch.arange(
                point_count, dtype=torch.int64, device=self._xyz.device
            )
        else:
            source_ids = source_ids.detach().to(
                device=self._xyz.device, dtype=torch.int64
            ).contiguous()
        if source_ids.ndim != 1 or source_ids.shape[0] != point_count:
            raise ValueError("source IDs must have one row per Gaussian")
        if origin_point_count is None:
            origin_point_count = point_count
        origin_point_count = int(origin_point_count)
        if source_ids.numel():
            if int(source_ids.min()) < 0 or int(source_ids.max()) >= origin_point_count:
                raise ValueError("source IDs are outside the origin row range")
            if source_ids.numel() > 1 and not bool(
                torch.all(source_ids[1:] > source_ids[:-1])
            ):
                raise ValueError(
                    "source IDs must be strictly increasing; rows were reordered or duplicated"
                )
        self._source_ids = source_ids
        self._source_id_origin_count = origin_point_count
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_language_feature_logits(self):
        if self._language_feature_logits is not None:
            return self._language_feature_logits
        else:
            raise ValueError('language feature logits is None')
    
    @property
    def get_language_feature_codebooks(self):
        if self._language_feature_codebooks is not None:
            return self._language_feature_codebooks
        else:
            raise ValueError('language feature codebooks is None')

    @property
    def semantic_level_count(self):
        """Number of semantic-scale heads stored in this Gaussian model."""
        if self._language_feature_logits is None:
            return 0
        return 1 if self._language_feature_logits.ndim == 2 else self._language_feature_logits.shape[1]

    def _semantic_logits(self):
        """Return logits as [N, semantic_level, RVQ_layer * codebook]."""
        logits = self.get_language_feature_logits
        return logits.unsqueeze(1) if logits.ndim == 2 else logits

    def _semantic_codebooks(self):
        """Return codebooks as [semantic_level, RVQ_layer, codebook, 512]."""
        codebooks = self.get_language_feature_codebooks
        return codebooks.unsqueeze(0) if codebooks.ndim == 3 else codebooks
    
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
        # language_feature = torch.zeros((fused_point_cloud.shape[0], 512), device="cuda")
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
        # 在从pointcloud初始化的时候是再训练原始gs的时候，这个时候不需要进行feature的初始化
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if training_args.include_feature:
            semantic_level_num = getattr(training_args, "semantic_level_num", 1)
            expected_logit_width = training_args.vq_layer_num * training_args.codebook_size
            semantic_shape_matches = (
                self._language_feature_logits is not None
                and self._language_feature_logits.shape[0] == self._xyz.shape[0]
                and self.semantic_level_count == semantic_level_num
                and self._semantic_logits().shape[-1] == expected_logit_width
                and self._semantic_codebooks().shape[1:3]
                == (training_args.vq_layer_num, training_args.codebook_size)
            )
            if not semantic_shape_matches:
                # initialize language feature logits and codebooks
                if semantic_level_num == 1:
                    language_feature_logits = torch.zeros(
                        (self._xyz.shape[0], expected_logit_width), device="cuda"
                    )
                    language_feature_codebooks = torch.randn(
                        (training_args.vq_layer_num, training_args.codebook_size, 512),
                        device="cuda",
                    )
                else:
                    language_feature_logits = torch.zeros(
                        (self._xyz.shape[0], semantic_level_num, expected_logit_width),
                        device="cuda",
                    )
                    language_feature_codebooks = torch.randn(
                        (
                            semantic_level_num,
                            training_args.vq_layer_num,
                            training_args.codebook_size,
                            512,
                        ),
                        device="cuda",
                    )
                self._language_feature_logits = nn.Parameter(language_feature_logits.requires_grad_(True))
                self._language_feature_codebooks = nn.Parameter(language_feature_codebooks.requires_grad_(True))

            # Keep logits and codebooks in separate groups: logits have one row
            # per Gaussian and must follow any pruning mask, while codebooks are
            # global and must never be point-pruned.
            language_groups = [
                {'params': [self._language_feature_logits],
                 'lr': training_args.language_feature_lr, "name": "language_feature_logits"},
                {'params': [self._language_feature_codebooks],
                 'lr': training_args.language_feature_lr, "name": "language_feature_codebooks"},
            ]
            support_groups = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]
            support_parameters = (
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
            )
            if getattr(training_args, "support_only", False):
                l = support_groups
                for parameter in support_parameters:
                    parameter.requires_grad_(True)
                self._language_feature_logits.requires_grad_(False)
                self._language_feature_codebooks.requires_grad_(False)
            elif getattr(training_args, "joint_optimize", False):
                l = support_groups + language_groups
                for parameter in support_parameters:
                    parameter.requires_grad_(True)
                self._language_feature_logits.requires_grad_(True)
                self._language_feature_codebooks.requires_grad_(True)
            else:
                l = language_groups
                for parameter in support_parameters:
                    parameter.requires_grad_(False)
                self._language_feature_logits.requires_grad_(True)
                self._language_feature_codebooks.requires_grad_(True)
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]
            assert self._language_feature_logits is None and self._language_feature_codebooks is None

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

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
        # l.append('language_feature')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

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
        print(self._xyz.shape[0])
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

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
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self._source_ids is not None:
            if valid_points_mask.shape[0] != self._source_ids.shape[0]:
                raise ValueError("pruning mask and source-ID rows disagree")
            next_source_ids = self._source_ids[valid_points_mask].contiguous()
        else:
            next_source_ids = None
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if next_source_ids is not None:
            self._source_ids = next_source_ids

    def _prune_named_point_parameter(self, name, parameter, valid_points_mask):
        """Prune one point-aligned parameter and its Adam state, if optimized."""
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            if len(group["params"]) != 1:
                raise RuntimeError("Optimizer group '{}' must contain one tensor".format(name))
            old_parameter = group["params"][0]
            stored_state = self.optimizer.state.pop(old_parameter, None)
            new_parameter = nn.Parameter(old_parameter[valid_points_mask].requires_grad_(True))
            group["params"][0] = new_parameter
            if stored_state is not None:
                for state_name, state_value in tuple(stored_state.items()):
                    if (
                        torch.is_tensor(state_value)
                        and state_value.ndim > 0
                        and state_value.shape[0] == valid_points_mask.shape[0]
                    ):
                        stored_state[state_name] = state_value[valid_points_mask]
                self.optimizer.state[new_parameter] = stored_state
            return new_parameter

        # A frozen point attribute is not present in the optimizer, but it must
        # still remain aligned with the shared Gaussian mask.
        return nn.Parameter(
            parameter[valid_points_mask], requires_grad=parameter.requires_grad
        )

    def prune_points_admm(self, mask):
        """Apply one shared mask to geometry and every semantic-scale head."""
        valid_points_mask = ~mask
        before = self._xyz.shape[0]
        if self._source_ids is not None:
            if valid_points_mask.shape[0] != self._source_ids.shape[0]:
                raise ValueError("ADMM pruning mask and source-ID rows disagree")
            next_source_ids = self._source_ids[valid_points_mask].contiguous()
        else:
            next_source_ids = None
        self._xyz = self._prune_named_point_parameter("xyz", self._xyz, valid_points_mask)
        self._features_dc = self._prune_named_point_parameter("f_dc", self._features_dc, valid_points_mask)
        self._features_rest = self._prune_named_point_parameter("f_rest", self._features_rest, valid_points_mask)
        self._opacity = self._prune_named_point_parameter("opacity", self._opacity, valid_points_mask)
        self._scaling = self._prune_named_point_parameter("scaling", self._scaling, valid_points_mask)
        self._rotation = self._prune_named_point_parameter("rotation", self._rotation, valid_points_mask)
        if self._language_feature_logits is not None:
            self._language_feature_logits = self._prune_named_point_parameter(
                "language_feature_logits",
                self._language_feature_logits,
                valid_points_mask,
            )

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if next_source_ids is not None:
            self._source_ids = next_source_ids
        print("Pruned Gaussians: {} -> {}".format(before, self._xyz.shape[0]))

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

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        if self._source_ids is not None and new_xyz.shape[0] > 0:
            raise RuntimeError(
                "Source-ID controlled runs prohibit densification because a new "
                "Gaussian has no unique RGB-host source row."
            )
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        # "language_feature": new_language_feature,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        self._xyz = optimizable_tensors["xyz"]
        # print(self._xyz.shape[0])
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

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
        # new_language_feature = self._language_feature[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

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
        # new_language_feature = self._language_feature[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

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
    
    def get_render_weights(self, k, semantic_level=0):
        logits = self._semantic_logits()[:, semantic_level]
        _, layer_num, codebook_size, _ = self._semantic_codebooks().shape
        weights = []
        for i in range(layer_num):
            soft_code = softmax_to_topk_soft_code(logits[:, i*codebook_size:(i+1)*codebook_size], k)
            weights.append(soft_code)
        return torch.cat(weights, dim=-1).float()
    
    def compute_feature_maps(self, language_feature_weight_map, semantic_level=0):
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(D, -1)
        language_features = []
        codebooks = self._semantic_codebooks()[semantic_level]
        layer_num, codebook_size, _ = codebooks.shape
        for i in range(layer_num):
            language_feature = codebooks[i].T @ language_feature_weight_map[i * codebook_size:(i+1)*codebook_size]
            language_feature = language_feature.view(512, H, W)
            if i > 0:
                language_feature += language_features[-1].detach()
            language_features.append(language_feature)
        return torch.stack(language_features, dim=1)

    def compute_layer_feature_map(self, language_feature_weight_map, layer_idx, semantic_level=0):
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(D, -1)
        codebooks = self._semantic_codebooks()[semantic_level]
        layer_num, codebook_size, _ = codebooks.shape
        if layer_idx < 0 or layer_idx >= layer_num:
            raise ValueError("layer_idx {} is outside [0, {})".format(layer_idx, layer_num))
        for i in range(layer_idx + 1):
            language_feature = codebooks[i].T @ language_feature_weight_map[i * codebook_size:(i+1)*codebook_size]
            language_feature = language_feature.view(512, H, W)
            if i > 0:
                language_feature += language_feature_before.detach()
            language_feature_before = language_feature
        return language_feature
    
    def compute_final_feature_map(self, language_feature_weight_map, semantic_level=0):
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(D, -1) 
        codebooks = self._semantic_codebooks()[semantic_level]
        language_feature = codebooks.reshape(-1, 512).T @ language_feature_weight_map
        language_feature = language_feature.view(512, H, W)
        return language_feature

    def prepare_multiscale_quick_render(self, topk=4):
        """Build the released 3-scale sparse quick-render representation."""
        logits = self._semantic_logits()
        codebooks = self._semantic_codebooks()
        semantic_levels, rvq_layers, codebook_size, feature_dim = codebooks.shape
        if semantic_levels != 3 or rvq_layers != 1:
            raise ValueError(
                "Quick rendering requires 3 semantic heads with one RVQ layer; "
                "got {} heads and {} layers".format(semantic_levels, rvq_layers)
            )
        if codebook_size != 64 or feature_dim != 512:
            raise ValueError(
                "The released quick rasterizer requires [3, 64, 512] codebooks"
            )
        weights, indices = [], []
        for semantic_level in range(semantic_levels):
            level_weights, level_indices = get_weights_and_indices(
                logits[:, semantic_level], topk
            )
            weights.append(level_weights)
            indices.append(level_indices + semantic_level * codebook_size)
        self._language_feature_weights = torch.cat(weights, dim=1)
        self._language_feature_indices = torch.cat(indices, dim=1)

    def get_quick_codebooks(self):
        """Return quick-render codebooks as [3, 64, 512]."""
        codebooks = self.get_language_feature_codebooks
        if codebooks.ndim == 4:
            if codebooks.shape[1] != 1:
                raise ValueError("Quick rendering only supports one RVQ layer")
            return codebooks[:, 0]
        return codebooks
