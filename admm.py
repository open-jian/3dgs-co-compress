"""ADMM pruning and SH projection adapted from CoLaSplat for LangSplatV2.

The semantic representation is already vector-quantized by LangSplatV2.  The
second ADMM constraint therefore remains on RGB spherical-harmonic attributes,
while the first constraint sparsifies opacity.  All semantic heads are pruned
later with exactly the same Gaussian mask by ``GaussianModel.prune_points_admm``.
"""

import torch


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


class SHKMeansProjector:
    """Chunked GPU k-means projection for one SH vector per Gaussian."""

    def __init__(self, num_clusters, chunk_size=4096):
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        self.requested_clusters = num_clusters
        self.chunk_size = chunk_size
        self.centers = None
        self.assignments = None
        self.feature_shape = None

    @staticmethod
    def _flatten(features):
        return features.detach().reshape(features.shape[0], -1).float()

    def _assign(self, features):
        assignments = []
        for start in range(0, features.shape[0], self.chunk_size):
            chunk = features[start:start + self.chunk_size]
            distances = torch.cdist(chunk, self.centers)
            assignments.append(distances.argmin(dim=1))
        return torch.cat(assignments, dim=0)

    def _update_centers(self, features, assignments):
        cluster_count = self.centers.shape[0]
        sums = torch.zeros_like(self.centers)
        sums.scatter_add_(
            0, assignments[:, None].expand(-1, features.shape[1]), features
        )
        counts = torch.bincount(assignments, minlength=cluster_count).to(features.dtype)
        nonempty = counts > 0
        updated = self.centers.clone()
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        self.centers = updated

    def project(self, features, update_centers=True):
        flattened = self._flatten(features)
        self.feature_shape = tuple(features.shape[1:])
        if self.centers is None:
            cluster_count = min(self.requested_clusters, flattened.shape[0])
            sample_indices = torch.randperm(
                flattened.shape[0], device=flattened.device
            )[:cluster_count]
            self.centers = flattened[sample_indices].clone()

        assignments = self._assign(flattened)
        if update_centers:
            self._update_centers(flattened, assignments)
            assignments = self._assign(flattened)
        self.assignments = assignments
        return self.centers[assignments].reshape_as(features)

    def prune(self, mask):
        if self.assignments is not None:
            self.assignments = self.assignments[~mask]

    def state_dict(self):
        if self.centers is None or self.assignments is None:
            raise RuntimeError("SH quantizer has not been initialized")
        return {
            "centers": self.centers.detach().cpu(),
            "indices": self.assignments.detach().to(torch.int32).cpu(),
            "feature_shape": self.feature_shape,
        }


class ADMM:
    """Two-constraint ADMM state for opacity sparsity and SH quantization."""

    def __init__(self, gaussian_model, rho_opacity, rho_sh, sh_clusters, device="cuda"):
        self.gaussian_model = gaussian_model
        self.device = torch.device(device)
        self.rho_opacity = rho_opacity
        self.rho_sh = rho_sh
        self.u_opacity = torch.zeros_like(gaussian_model.get_opacity, device=self.device)
        self.z_opacity = gaussian_model.get_opacity.detach().clone()
        self.u_sh = torch.zeros_like(gaussian_model._features_rest, device=self.device)
        self.z_sh = gaussian_model._features_rest.detach().clone()
        self.sh_projector = SHKMeansProjector(sh_clusters)

    @torch.no_grad()
    def update_opacity(self, pruning_fraction, update_dual=True):
        value = self.gaussian_model.get_opacity + self.u_opacity
        prune_mask = get_pruning_mask(value[:, 0], pruning_fraction)
        self.z_opacity = value.clone()
        self.z_opacity[prune_mask] = 0
        if update_dual:
            self.u_opacity.add_(self.gaussian_model.get_opacity - self.z_opacity)

    @torch.no_grad()
    def update_sh(self, update_dual=True, update_centers=True):
        value = self.gaussian_model._features_rest + self.u_sh
        self.z_sh = self.sh_projector.project(value, update_centers=update_centers)
        if update_dual:
            self.u_sh.add_(self.gaussian_model._features_rest - self.z_sh)

    def opacity_loss(self):
        residual = self.gaussian_model.get_opacity - self.z_opacity + self.u_opacity
        return 0.5 * self.rho_opacity * residual.square().sum()

    def sh_loss(self):
        residual = self.gaussian_model._features_rest - self.z_sh + self.u_sh
        return 0.5 * self.rho_sh * residual.square().sum()

    @torch.no_grad()
    def prune(self, mask):
        valid = ~mask
        self.u_opacity = self.u_opacity[valid]
        self.z_opacity = self.z_opacity[valid]
        self.u_sh = self.u_sh[valid]
        self.z_sh = self.z_sh[valid]
        self.sh_projector.prune(mask)

