"""Joint opacity sparsification and C3DGS-style attribute VQ via ADMM."""

from __future__ import annotations

from typing import Dict, List

import torch

from c3dgs_quantization import (
    QUANTIZATION_FORMAT,
    QUANTIZATION_VERSION,
    C3DGSSensitivityProjector,
    covariance_to_rotation_scale,
    normalized_covariance,
)


def get_pruning_mask(scores, pruning_fraction):
    """Return an exact-size mask for the lowest-scoring Gaussian points."""
    if not 0.0 <= pruning_fraction < 1.0:
        raise ValueError("pruning_fraction must be in [0, 1)")
    scores = scores.reshape(-1)
    prune_count = int(pruning_fraction * scores.numel())
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if prune_count:
        indices = torch.topk(scores, prune_count, largest=False, sorted=False).indices
        mask[indices] = True
    return mask


class ADMM:
    """ADMM state for shared support and block-wise C3DGS vector quantization.

    The VQ blocks mirror the useful structure of C3DGS: one color vector
    (DC plus directional SH) and one normalized covariance vector per Gaussian.
    ClunoGS adds one independent simplex-constrained coefficient block per
    semantic level. XYZ and opacity never enter vector quantization.
    """

    def __init__(
        self,
        gaussian_model,
        rho_opacity,
        rho_color,
        color_clusters,
        *,
        rho_covariance=None,
        rho_semantic=None,
        covariance_clusters=None,
        semantic_clusters=None,
        semantic_topk=4,
        codebook_decay=0.8,
        sensitivity_decay=0.9,
        keep_ratio=0.0,
        refinement_steps=1,
        covariance_refinement_steps=None,
        semantic_refinement_steps=None,
        chunk_size=4096,
        enable_attribute_vq=True,
        device=None,
    ):
        if not 0.0 <= sensitivity_decay < 1.0:
            raise ValueError("sensitivity_decay must be in [0, 1)")
        self.gaussian_model = gaussian_model
        if device is None:
            device = gaussian_model.get_xyz.device
        self.device = torch.device(device)
        self.rho_opacity = float(rho_opacity)
        self.rho_color = float(rho_color)
        self.rho_covariance = float(
            rho_color if rho_covariance is None else rho_covariance
        )
        self.rho_semantic = float(
            rho_color if rho_semantic is None else rho_semantic
        )
        self.sensitivity_decay = float(sensitivity_decay)
        covariance_clusters = (
            color_clusters if covariance_clusters is None else covariance_clusters
        )
        semantic_clusters = (
            color_clusters if semantic_clusters is None else semantic_clusters
        )
        covariance_refinement_steps = (
            refinement_steps
            if covariance_refinement_steps is None
            else covariance_refinement_steps
        )
        semantic_refinement_steps = (
            refinement_steps
            if semantic_refinement_steps is None
            else semantic_refinement_steps
        )

        self.u_opacity = torch.zeros_like(
            gaussian_model.get_opacity, device=self.device
        )
        self.z_opacity = gaussian_model.get_opacity.detach().clone()
        self.opacity_prune_mask = torch.zeros(
            gaussian_model.get_xyz.shape[0], dtype=torch.bool, device=self.device
        )

        self.attribute_vq_enabled = bool(enable_attribute_vq)
        point_count = int(gaussian_model.get_xyz.shape[0])
        self.semantic_topk = int(semantic_topk)
        self.semantic_projectors: List[C3DGSSensitivityProjector] = []
        self.u_semantic: List[torch.Tensor] = []
        self.z_semantic: List[torch.Tensor] = []
        self.semantic_sensitivity: List[torch.Tensor] = []
        self._sensitivity_observations = 0
        if not self.attribute_vq_enabled:
            self.color_projector = None
            self.covariance_projector = None
            return

        color = self._current_color()
        covariance = self._current_covariance()
        self.u_color = torch.zeros_like(color, device=self.device)
        self.z_color = color.detach().clone()
        self.u_covariance = torch.zeros_like(covariance, device=self.device)
        self.z_covariance = covariance.detach().clone()
        self.color_projector = C3DGSSensitivityProjector(
            color_clusters,
            chunk_size=chunk_size,
            decay=codebook_decay,
            keep_ratio=keep_ratio,
            refinement_steps=refinement_steps,
            constraint="none",
        )
        self.covariance_projector = C3DGSSensitivityProjector(
            covariance_clusters,
            chunk_size=chunk_size,
            decay=codebook_decay,
            keep_ratio=keep_ratio,
            refinement_steps=covariance_refinement_steps,
            constraint="covariance",
        )

        self.color_sensitivity = torch.zeros(point_count, device=self.device)
        self.covariance_sensitivity = torch.zeros(point_count, device=self.device)
        if gaussian_model._language_feature_logits is not None:
            gaussian_model.freeze_semantic_indices(self.semantic_topk)
            semantic = self._current_semantic()
            for semantic_level in range(semantic.shape[1]):
                level_values = semantic[:, semantic_level]
                self.semantic_projectors.append(
                    C3DGSSensitivityProjector(
                        semantic_clusters,
                        chunk_size=chunk_size,
                        decay=codebook_decay,
                        keep_ratio=keep_ratio,
                        refinement_steps=semantic_refinement_steps,
                        constraint="simplex",
                    )
                )
                self.u_semantic.append(torch.zeros_like(level_values))
                self.z_semantic.append(level_values.detach().clone())
                self.semantic_sensitivity.append(
                    torch.zeros(point_count, device=self.device)
                )

    def _current_color(self):
        return self.gaussian_model.get_features

    def _current_covariance(self):
        covariance, _ = normalized_covariance(self.gaussian_model)
        return covariance

    def _current_semantic(self):
        return self.gaussian_model.get_semantic_coefficients()

    def _update_sensitivity(self, current, observation):
        observation = observation.detach().reshape(-1).to(current)
        observation = observation.abs().nan_to_num(0.0, posinf=0.0, neginf=0.0)
        if self._sensitivity_observations == 0:
            current.copy_(observation)
        else:
            current.mul_(self.sensitivity_decay).add_(
                observation, alpha=1.0 - self.sensitivity_decay
            )

    @staticmethod
    def _gradient_importance(parameters):
        flattened = []
        point_count = None
        device = None
        for parameter in parameters:
            if parameter is None:
                continue
            point_count = int(parameter.shape[0])
            device = parameter.device
            if parameter.grad is not None:
                flattened.append(
                    parameter.grad.detach().reshape(point_count, -1).abs()
                )
        if point_count is None:
            raise ValueError("at least one point-aligned parameter is required")
        if not flattened:
            return torch.zeros(point_count, device=device)
        return torch.cat(flattened, dim=1).amax(dim=1)

    @torch.no_grad()
    def observe_sensitivity(self):
        """Update C3DGS sensitivity EMAs from the current task gradients."""
        if not self.attribute_vq_enabled:
            return
        color_importance = self._gradient_importance(
            (
                self.gaussian_model._features_dc,
                self.gaussian_model._features_rest,
            )
        )
        covariance_importance = self._gradient_importance(
            (self.gaussian_model._scaling, self.gaussian_model._rotation)
        )
        self._update_sensitivity(self.color_sensitivity, color_importance)
        self._update_sensitivity(
            self.covariance_sensitivity, covariance_importance
        )

        if self.semantic_projectors:
            logits = self.gaussian_model._semantic_logits()
            codebooks = self.gaussian_model._semantic_codebooks()
            rvq_layers, codebook_size = codebooks.shape[1:3]
            if self.gaussian_model._language_feature_logits.grad is None:
                semantic_grad = torch.zeros_like(logits)
            else:
                semantic_grad = (
                    self.gaussian_model._language_feature_logits.grad.detach().abs()
                )
                if semantic_grad.ndim == 2:
                    semantic_grad = semantic_grad.unsqueeze(1)
            semantic_grad = semantic_grad.reshape(
                logits.shape[0], logits.shape[1], rvq_layers, codebook_size
            )
            fixed_indices = self.gaussian_model._semantic_fixed_indices
            selected = torch.gather(semantic_grad, -1, fixed_indices)
            for level, level_sensitivity in enumerate(self.semantic_sensitivity):
                self._update_sensitivity(
                    level_sensitivity,
                    selected[:, level]
                    .reshape(selected.shape[0], -1)
                    .amax(dim=1),
                )
        self._sensitivity_observations += 1

    @torch.no_grad()
    def update_opacity(self, pruning_fraction, update_dual=True):
        value = self.gaussian_model.get_opacity + self.u_opacity
        self.opacity_prune_mask = get_pruning_mask(value[:, 0], pruning_fraction)
        self.z_opacity = value.clone()
        self.z_opacity[self.opacity_prune_mask] = 0
        if update_dual:
            self.u_opacity.add_(
                self.gaussian_model.get_opacity - self.z_opacity
            )

    @torch.no_grad()
    def update_attributes(self, update_dual=True, update_codebooks=True):
        if not self.attribute_vq_enabled:
            raise RuntimeError("C3DGS attribute VQ is disabled")
        color = self._current_color()
        color_value = color + self.u_color
        self.z_color = self.color_projector.project(
            color_value,
            self.color_sensitivity,
            update_codebook=update_codebooks,
        )
        if update_dual:
            self.u_color.add_(color - self.z_color)

        covariance = self._current_covariance()
        covariance_value = covariance + self.u_covariance
        self.z_covariance = self.covariance_projector.project(
            covariance_value,
            self.covariance_sensitivity,
            update_codebook=update_codebooks,
        )
        if update_dual:
            self.u_covariance.add_(covariance - self.z_covariance)

        if self.semantic_projectors:
            semantic = self._current_semantic()
            for level, projector in enumerate(self.semantic_projectors):
                level_values = semantic[:, level]
                corrected = level_values + self.u_semantic[level]
                self.z_semantic[level] = projector.project(
                    corrected,
                    self.semantic_sensitivity[level],
                    update_codebook=update_codebooks,
                )
                if update_dual:
                    self.u_semantic[level].add_(
                        level_values - self.z_semantic[level]
                    )

    def opacity_loss(self):
        residual = (
            self.gaussian_model.get_opacity - self.z_opacity + self.u_opacity
        )
        return 0.5 * self.rho_opacity * residual.square().sum()

    def attribute_losses(self) -> Dict[str, torch.Tensor]:
        if not self.attribute_vq_enabled:
            zero = self.gaussian_model.get_opacity.new_zeros(())
            return {"color": zero, "covariance": zero, "semantic": zero}
        color_residual = self._current_color() - self.z_color + self.u_color
        covariance_residual = (
            self._current_covariance()
            - self.z_covariance
            + self.u_covariance
        )
        color_loss = 0.5 * self.rho_color * color_residual.square().sum()
        covariance_loss = (
            0.5 * self.rho_covariance * covariance_residual.square().sum()
        )
        if self.semantic_projectors:
            semantic = self._current_semantic()
            semantic_loss = semantic.new_zeros(())
            for level in range(len(self.semantic_projectors)):
                residual = (
                    semantic[:, level]
                    - self.z_semantic[level]
                    + self.u_semantic[level]
                )
                semantic_loss = semantic_loss + (
                    0.5 * self.rho_semantic * residual.square().sum()
                )
        else:
            semantic_loss = color_loss.new_zeros(())
        return {
            "color": color_loss,
            "covariance": covariance_loss,
            "semantic": semantic_loss,
        }

    @torch.no_grad()
    def materialize_attributes(self):
        """Replace trainable attributes with their final indexed projections."""
        if not self.attribute_vq_enabled:
            raise RuntimeError("C3DGS attribute VQ is disabled")
        color = self.color_projector.project(
            self._current_color(),
            self.color_sensitivity,
            update_codebook=False,
        )
        dc_width = self.gaussian_model._features_dc.shape[1]
        self.gaussian_model._features_dc.copy_(color[:, :dc_width])
        self.gaussian_model._features_rest.copy_(color[:, dc_width:])

        covariance = self.covariance_projector.project(
            self._current_covariance(),
            self.covariance_sensitivity,
            update_codebook=False,
        )
        rotation, normalized_scale = covariance_to_rotation_scale(covariance)
        _, scale_factor = normalized_covariance(self.gaussian_model)
        scaling = (normalized_scale * scale_factor).clamp_min(1e-12).log()
        self.gaussian_model._scaling.copy_(scaling)
        self.gaussian_model._rotation.copy_(rotation)

        if self.semantic_projectors:
            semantic = self._current_semantic()
            projected_levels = []
            for level, projector in enumerate(self.semantic_projectors):
                projected_levels.append(
                    projector.project(
                        semantic[:, level],
                        self.semantic_sensitivity[level],
                        update_codebook=False,
                    )
                )
            self.gaussian_model.materialize_semantic_coefficients(
                torch.stack(projected_levels, dim=1)
            )

    @torch.no_grad()
    def quantization_state(self) -> Dict[str, object]:
        """Return a deployment-oriented, point-aligned C3DGS VQ sidecar."""
        if not self.attribute_vq_enabled:
            raise RuntimeError("C3DGS attribute VQ is disabled")
        color = self._current_color()
        covariance, scale_factor = normalized_covariance(self.gaussian_model)
        blocks = {
            "color": self.color_projector.encode(
                color, self.color_sensitivity
            ),
            "covariance": self.covariance_projector.encode(
                covariance, self.covariance_sensitivity
            ),
        }
        if self.semantic_projectors:
            semantic = self._current_semantic()
            semantic_blocks = []
            for level, projector in enumerate(self.semantic_projectors):
                semantic_blocks.append(
                    projector.encode(
                        semantic[:, level], self.semantic_sensitivity[level]
                    )
                )
            blocks["semantic"] = semantic_blocks
        return {
            "format": QUANTIZATION_FORMAT,
            "version": QUANTIZATION_VERSION,
            "point_count": int(color.shape[0]),
            "algorithm": {
                "name": "C3DGS sensitivity-aware VQ adapted as ADMM projection",
                "assignment": "euclidean_nearest_codeword",
                "centroid_update": "sensitivity_weighted_ema",
                "sensitivity_source": "joint_task_gradient_ema",
            },
            "blocks": blocks,
            "scale_factor": scale_factor.detach().cpu().contiguous(),
            "semantic_fixed_indices": (
                None
                if self.gaussian_model._semantic_fixed_indices is None
                else self.gaussian_model._semantic_fixed_indices.detach()
                .to(torch.int16)
                .cpu()
                .contiguous()
            ),
            "semantic_topk": self.semantic_topk,
        }

    @torch.no_grad()
    def prune(self, mask):
        valid = ~mask
        self.u_opacity = self.u_opacity[valid]
        self.z_opacity = self.z_opacity[valid]
        self.opacity_prune_mask = self.opacity_prune_mask[valid]
        if not self.attribute_vq_enabled:
            return
        self.u_color = self.u_color[valid]
        self.z_color = self.z_color[valid]
        self.u_covariance = self.u_covariance[valid]
        self.z_covariance = self.z_covariance[valid]
        self.color_sensitivity = self.color_sensitivity[valid]
        self.covariance_sensitivity = self.covariance_sensitivity[valid]
        self.color_projector.prune(mask)
        self.covariance_projector.prune(mask)
        for level, projector in enumerate(self.semantic_projectors):
            self.u_semantic[level] = self.u_semantic[level][valid]
            self.z_semantic[level] = self.z_semantic[level][valid]
            self.semantic_sensitivity[level] = self.semantic_sensitivity[level][valid]
            projector.prune(mask)
