import tempfile
import unittest
from pathlib import Path

import torch

from c3dgs_quantization import (
    QUANTIZATION_FORMAT,
    QUANTIZATION_VERSION,
    C3DGSSensitivityProjector,
    covariance_to_rotation_scale,
    project_simplex,
    project_unit_trace_covariance,
    to_full_covariance,
)
from compact_artifact import (
    export_compact_artifact,
    load_compact_gaussians,
    validate_artifact,
)


class C3DGSProjectorTest(unittest.TestCase):
    def test_sensitivity_weights_centroid_refresh(self):
        projector = C3DGSSensitivityProjector(1, decay=0.0)
        projector.centers = torch.tensor([[5.0]])
        values = torch.tensor([[0.0], [10.0]])
        projected = projector.project(
            values, torch.tensor([100.0, 1.0]), update_codebook=True
        )
        expected = torch.tensor(10.0 / 101.0)
        self.assertTrue(torch.allclose(projector.centers[0, 0], expected))
        self.assertTrue(torch.allclose(projected[:, 0], expected.expand(2)))

    def test_sensitive_tail_is_stored_in_indexed_table(self):
        projector = C3DGSSensitivityProjector(
            1, decay=0.0, keep_ratio=0.25
        )
        projector.centers = torch.tensor([[1.0]])
        values = torch.tensor([[0.0], [1.0], [2.0], [9.0]])
        importance = torch.tensor([0.0, 0.0, 0.0, 10.0])
        projector.project(values, importance, update_codebook=True)
        encoded = projector.encode(values, importance)
        self.assertEqual(encoded["kept_count"], 1)
        self.assertEqual(encoded["values"].shape[0], 2)
        self.assertEqual(int(encoded["indices"][-1]), 1)
        self.assertEqual(float(encoded["values"][-1, 0]), 9.0)

    def test_constraints_produce_valid_values(self):
        simplex = project_simplex(
            torch.tensor([[2.0, -1.0, 0.5], [-2.0, -1.0, -3.0]])
        )
        self.assertTrue(bool((simplex >= 0).all()))
        self.assertTrue(torch.allclose(simplex.sum(-1), torch.ones(2)))

        packed = torch.tensor(
            [[2.0, 0.3, -0.4, -1.0, 0.2, 0.1]], dtype=torch.float32
        )
        projected = project_unit_trace_covariance(packed)
        matrix = to_full_covariance(projected)
        eigenvalues = torch.linalg.eigvalsh(matrix)
        self.assertTrue(bool((eigenvalues > 0).all()))
        self.assertTrue(torch.allclose(torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1), torch.ones(1)))
        quaternion, scale = covariance_to_rotation_scale(projected)
        self.assertTrue(torch.allclose(quaternion.norm(dim=-1), torch.ones(1)))
        self.assertTrue(bool((scale > 0).all()))


class CompactArtifactV2Test(unittest.TestCase):
    def _fixture(self, root):
        point_count = 5
        semantic_levels = 3
        semantic_atoms = 64
        topk = 4
        xyz = torch.arange(point_count * 3, dtype=torch.float32).reshape(point_count, 3)
        features_dc = torch.randn(point_count, 1, 3)
        features_rest = torch.randn(point_count, 15, 3)
        scaling = torch.zeros(point_count, 3)
        rotation = torch.zeros(point_count, 4)
        rotation[:, 0] = 1
        opacity = torch.zeros(point_count, 1)
        semantic_logits = torch.randn(point_count, semantic_levels, semantic_atoms)
        semantic_codebooks = torch.randn(
            semantic_levels, 1, semantic_atoms, 512
        )
        model_params = (
            3,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            semantic_logits,
            semantic_codebooks,
            torch.zeros(point_count),
            torch.zeros(point_count, 1),
            torch.zeros(point_count, 1),
            {},
            1.0,
        )
        checkpoint = root / "checkpoint.pth"
        torch.save((model_params, 1000), checkpoint)

        color_values = torch.randn(2, 48)
        covariance_values = torch.tensor(
            [
                [1 / 3, 0, 0, 1 / 3, 0, 1 / 3],
                [0.5, 0, 0, 0.25, 0, 0.25],
            ],
            dtype=torch.float32,
        )
        semantic_blocks = []
        for _ in range(semantic_levels):
            semantic_blocks.append(
                {
                    "values": torch.tensor(
                        [[0.7, 0.1, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1]]
                    ),
                    "indices": torch.arange(point_count, dtype=torch.int32) % 2,
                    "feature_shape": (1, topk),
                }
            )
        fixed_indices = torch.arange(topk, dtype=torch.int16).view(1, 1, 1, topk)
        fixed_indices = fixed_indices.expand(
            point_count, semantic_levels, 1, topk
        ).contiguous()
        quantization = {
            "format": QUANTIZATION_FORMAT,
            "version": QUANTIZATION_VERSION,
            "point_count": point_count,
            "semantic_topk": topk,
            "algorithm": {"name": "test"},
            "blocks": {
                "color": {
                    "values": color_values,
                    "indices": torch.arange(point_count, dtype=torch.int32) % 2,
                    "feature_shape": (16, 3),
                },
                "covariance": {
                    "values": covariance_values,
                    "indices": torch.arange(point_count, dtype=torch.int32) % 2,
                    "feature_shape": (6,),
                },
                "semantic": semantic_blocks,
            },
            "scale_factor": torch.ones(point_count, 1),
            "semantic_fixed_indices": fixed_indices,
        }
        sidecar = root / "quantization.pth"
        torch.save(quantization, sidecar)
        return checkpoint, sidecar, topk

    def test_export_validate_and_decode_v2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, sidecar, topk = self._fixture(root)
            artifact = root / "compact.pth"
            manifest = export_compact_artifact(
                checkpoint, sidecar, artifact, topk=topk
            )
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(manifest["attribute_mode"], "c3dgs_admm_vq")
            validation = validate_artifact(artifact)
            self.assertEqual(validation["point_count"], 5)
            self.assertEqual(validation["attribute_mode"], "c3dgs_admm_vq")
            model, bundle = load_compact_gaussians(artifact, device="cpu")
            self.assertEqual(tuple(model.get_features.shape), (5, 16, 3))
            self.assertEqual(tuple(model.get_scaling.shape), (5, 3))
            self.assertEqual(tuple(model._language_feature_weights.shape), (5, 12))
            self.assertEqual(bundle["version"], 2)


if __name__ == "__main__":
    unittest.main()
