"""C3DGS-style sensitivity-aware vector quantization for ADMM.

The original C3DGS quantizer assigns vectors by Euclidean distance and updates
the codebook with sensitivity-weighted centroids.  This module keeps those
semantics while exposing a deterministic, chunked projection suitable for an
ADMM auxiliary-variable update.  It intentionally has no dependency on the
C3DGS custom CUDA extension so that the projector can be tested on CPU and run
inside the LangSplatV2 environment.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


QUANTIZATION_FORMAT = "clunogs.c3dgs-admm-vq"
QUANTIZATION_VERSION = 1


def _strip_symmetric(matrix: torch.Tensor) -> torch.Tensor:
    """Pack a symmetric 3x3 matrix as (xx, xy, xz, yy, yz, zz)."""
    return torch.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 0, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 1],
            matrix[..., 1, 2],
            matrix[..., 2, 2],
        ),
        dim=-1,
    )


def to_full_covariance(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (..., 6) covariance vectors into symmetric (..., 3, 3)."""
    if packed.shape[-1] != 6:
        raise ValueError("packed covariance must have six channels")
    matrix = packed.new_zeros(*packed.shape[:-1], 3, 3)
    matrix[..., 0, 0] = packed[..., 0]
    matrix[..., 0, 1] = packed[..., 1]
    matrix[..., 1, 0] = packed[..., 1]
    matrix[..., 0, 2] = packed[..., 2]
    matrix[..., 2, 0] = packed[..., 2]
    matrix[..., 1, 1] = packed[..., 3]
    matrix[..., 1, 2] = packed[..., 4]
    matrix[..., 2, 1] = packed[..., 4]
    matrix[..., 2, 2] = packed[..., 5]
    return matrix


def normalized_covariance(gaussian_model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return C3DGS normalized covariance and the unclustered scale factor.

    C3DGS clusters a unit-trace covariance while retaining one scalar scale
    factor per Gaussian.  ClunoGS stores three log-scales, so this function
    performs that factorization without changing the renderer representation.
    """
    scaling = gaussian_model.get_scaling
    scale_factor = torch.linalg.vector_norm(scaling, dim=-1, keepdim=True)
    normalized_scaling = scaling / scale_factor.clamp_min(1e-12)
    covariance = gaussian_model.covariance_activation(
        normalized_scaling, 1.0, gaussian_model._rotation
    )
    return covariance, scale_factor


def project_unit_trace_covariance(packed: torch.Tensor) -> torch.Tensor:
    """Map packed matrices to the valid SPD, unit-trace covariance set."""
    original_shape = packed.shape
    matrix = to_full_covariance(packed.reshape(-1, 6).float())
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # Leave a float32 safety margin: a 1e-8 eigenvalue can become slightly
    # negative after packing and reconstructing the six symmetric entries.
    eigenvalues = eigenvalues.clamp_min(1e-6)
    eigenvalues = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True)
    projected = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)
    return _strip_symmetric(projected).to(packed.dtype).reshape(original_shape)


def project_simplex(values: torch.Tensor) -> torch.Tensor:
    """Project the last dimension onto the probability simplex."""
    original_shape = values.shape
    flat = values.reshape(-1, values.shape[-1]).float()
    sorted_values, _ = torch.sort(flat, dim=-1, descending=True)
    cumulative = sorted_values.cumsum(dim=-1) - 1.0
    ranks = torch.arange(
        1, flat.shape[-1] + 1, device=flat.device, dtype=flat.dtype
    ).view(1, -1)
    support = sorted_values - cumulative / ranks > 0
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    threshold = cumulative.gather(1, support_size - 1) / support_size.to(flat.dtype)
    projected = (flat - threshold).clamp_min(0)
    projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return projected.to(values.dtype).reshape(original_shape)


def covariance_to_rotation_scale(
    packed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode normalized packed covariance into quaternion and scale vectors."""
    matrix = to_full_covariance(packed.float())
    identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    eigenvalues, rotation_matrix = torch.linalg.eigh(matrix + identity * 1e-8)
    normalized_scale = eigenvalues.clamp_min(1e-12).sqrt()
    determinant = torch.linalg.det(rotation_matrix)
    rotation_matrix = rotation_matrix * determinant[..., None, None]
    quaternion = matrix_to_quaternion(rotation_matrix)
    quaternion = F.normalize(quaternion, dim=-1)
    return quaternion.to(packed.dtype), normalized_scale.to(packed.dtype)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to real-first quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape (..., 3, 3)")
    batch_shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_shape + (9,)), dim=-1
    )
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                (
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    floor = q_abs.new_tensor(0.1)
    candidates = candidates / (2.0 * q_abs[..., None].clamp_min(floor))
    selector = F.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(torch.bool)
    return candidates[selector].reshape(batch_shape + (4,))


class C3DGSSensitivityProjector:
    """Sensitivity-aware vector-codebook projection adapted from C3DGS.

    Nearest-neighbour assignments remain Euclidean, matching C3DGS.  Point
    sensitivity weights the centroid refresh.  A configurable most-sensitive
    tail can bypass vector clustering, as in C3DGS's ``importance_include``
    path; those rows are still represented through the same indexed table.
    """

    VALID_CONSTRAINTS = ("none", "covariance", "simplex")

    def __init__(
        self,
        num_clusters: int,
        *,
        chunk_size: int = 4096,
        decay: float = 0.8,
        keep_ratio: float = 0.0,
        refinement_steps: int = 1,
        constraint: str = "none",
    ) -> None:
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if not 0.0 <= keep_ratio < 1.0:
            raise ValueError("keep_ratio must be in [0, 1)")
        if refinement_steps <= 0:
            raise ValueError("refinement_steps must be positive")
        if constraint not in self.VALID_CONSTRAINTS:
            raise ValueError("unsupported codebook constraint: {}".format(constraint))
        self.requested_clusters = int(num_clusters)
        self.chunk_size = int(chunk_size)
        self.decay = float(decay)
        self.keep_ratio = float(keep_ratio)
        self.refinement_steps = int(refinement_steps)
        self.constraint = constraint
        self.centers: Optional[torch.Tensor] = None
        self.assignments: Optional[torch.Tensor] = None
        self.keep_mask: Optional[torch.Tensor] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None

    @staticmethod
    def _flatten(features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2:
            raise ValueError("features must have one point dimension and one feature dimension")
        return features.detach().reshape(features.shape[0], -1).float()

    def _constrain(self, values: torch.Tensor) -> torch.Tensor:
        if self.constraint == "covariance":
            return project_unit_trace_covariance(values)
        if self.constraint == "simplex":
            return project_simplex(values)
        return values

    @staticmethod
    def _normalized_importance(
        importance: Optional[torch.Tensor], point_count: int, reference: torch.Tensor
    ) -> torch.Tensor:
        if importance is None:
            return reference.new_ones(point_count)
        importance = importance.detach().reshape(-1).to(
            device=reference.device, dtype=reference.dtype
        )
        if importance.shape[0] != point_count:
            raise ValueError("importance must contain one value per point")
        importance = importance.abs().nan_to_num(0.0, posinf=0.0, neginf=0.0)
        maximum = importance.max()
        if not bool(maximum > 0):
            return torch.ones_like(importance)
        return importance / maximum

    def _select_keep_mask(self, importance: torch.Tensor) -> torch.Tensor:
        keep_count = int(math.ceil(self.keep_ratio * importance.numel()))
        # Always leave at least one row for the clustered codebook.  This only
        # matters for tiny smoke/unit-test tensors; real scenes contain many
        # thousands of Gaussians.
        keep_count = min(keep_count, max(importance.numel() - 1, 0))
        mask = torch.zeros_like(importance, dtype=torch.bool)
        if keep_count:
            indices = torch.topk(
                importance, keep_count, largest=True, sorted=False
            ).indices
            mask[indices] = True
        return mask

    def _initialize_centers(self, features: torch.Tensor) -> None:
        cluster_count = min(self.requested_clusters, features.shape[0])
        if cluster_count <= 0:
            self.centers = features.new_empty((0, features.shape[1]))
            return
        # C3DGS initializes uniformly over the observed value range.  The same
        # scalar range is used for every channel in the released implementation.
        minimum, maximum = torch.aminmax(features)
        if bool(maximum > minimum):
            centers = torch.rand(
                cluster_count,
                features.shape[1],
                device=features.device,
                dtype=features.dtype,
            )
            centers = centers * (maximum - minimum) + minimum
        else:
            centers = features[:1].expand(cluster_count, -1).clone()
        self.centers = self._constrain(centers)

    def _assign(self, features: torch.Tensor) -> torch.Tensor:
        if self.centers is None or self.centers.shape[0] == 0:
            raise RuntimeError("C3DGS codebook is empty")
        assignments = []
        for start in range(0, features.shape[0], self.chunk_size):
            chunk = features[start:start + self.chunk_size]
            distances = torch.cdist(chunk, self.centers)
            assignments.append(distances.argmin(dim=1))
        return torch.cat(assignments, dim=0)

    def _refresh_centers(
        self,
        features: torch.Tensor,
        assignments: torch.Tensor,
        importance: torch.Tensor,
    ) -> None:
        cluster_count = self.centers.shape[0]
        weighted_features = features * importance[:, None]
        sums = torch.zeros_like(self.centers)
        sums.scatter_add_(
            0,
            assignments[:, None].expand(-1, features.shape[1]),
            weighted_features,
        )
        weights = features.new_zeros(cluster_count)
        weights.scatter_add_(0, assignments, importance)
        nonempty = weights > 0
        refreshed = self.centers.clone()
        refreshed[nonempty] = sums[nonempty] / weights[nonempty, None]
        self.centers = self._constrain(
            self.centers * self.decay + refreshed * (1.0 - self.decay)
        )

    def project(
        self,
        features: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        *,
        update_codebook: bool = True,
    ) -> torch.Tensor:
        flattened = self._flatten(features)
        self.feature_shape = tuple(features.shape[1:])
        sensitivity = self._normalized_importance(
            importance, flattened.shape[0], flattened
        )
        if update_codebook or self.keep_mask is None:
            self.keep_mask = self._select_keep_mask(sensitivity)
        elif self.keep_mask.shape[0] != flattened.shape[0]:
            raise ValueError("stored C3DGS keep mask and feature rows disagree")

        vector_mask = ~self.keep_mask
        vector_features = flattened[vector_mask]
        vector_importance = sensitivity[vector_mask]
        projected = flattened.clone()
        assignments = torch.full(
            (flattened.shape[0],), -1, device=flattened.device, dtype=torch.long
        )
        if vector_features.shape[0]:
            if self.centers is None:
                self._initialize_centers(vector_features)
            if update_codebook:
                for _ in range(self.refinement_steps):
                    vector_assignments = self._assign(vector_features)
                    self._refresh_centers(
                        vector_features, vector_assignments, vector_importance
                    )
            vector_assignments = self._assign(vector_features)
            assignments[vector_mask] = vector_assignments
            projected[vector_mask] = self.centers[vector_assignments]
        self.assignments = assignments
        return projected.reshape_as(features).to(features.dtype)

    def encode(
        self,
        features: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Encode points as one C3DGS table plus a per-point table index."""
        self.project(features, importance, update_codebook=False)
        flattened = self._flatten(features)
        if self.keep_mask is None or self.assignments is None or self.centers is None:
            raise RuntimeError("C3DGS projector has not been initialized")
        kept_values = self._constrain(flattened[self.keep_mask])
        table = torch.cat((self.centers, kept_values), dim=0)
        indices = self.assignments.clone()
        kept_count = int(self.keep_mask.sum())
        if kept_count:
            indices[self.keep_mask] = torch.arange(
                self.centers.shape[0],
                self.centers.shape[0] + kept_count,
                device=indices.device,
                dtype=indices.dtype,
            )
        if indices.numel() and int(indices.min()) < 0:
            raise RuntimeError("C3DGS encoding contains an unassigned row")
        return {
            "algorithm": "c3dgs_sensitivity_aware_vq",
            "values": table.detach().cpu().contiguous(),
            "indices": indices.detach().to(torch.int32).cpu().contiguous(),
            "feature_shape": self.feature_shape,
            "codebook_size": int(self.centers.shape[0]),
            "kept_count": kept_count,
            "keep_ratio": self.keep_ratio,
            "decay": self.decay,
            "constraint": self.constraint,
        }

    def prune(self, mask: torch.Tensor) -> None:
        if self.assignments is not None:
            self.assignments = self.assignments[~mask]
        if self.keep_mask is not None:
            self.keep_mask = self.keep_mask[~mask]
