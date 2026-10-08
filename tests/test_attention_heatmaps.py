import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from svoo.attention_heatmaps import AttentionHeatmaps, cluster_boundaries, pooled_scores
from svoo.models.wan.npu_attention import WanNPUProcessor


class HeatmapTests(unittest.TestCase):
    def test_pooled_scores_match_full_score_block_means(self):
        generator = torch.Generator().manual_seed(21)
        q = torch.randn(11, 8, generator=generator).bfloat16()
        k = torch.randn(11, 8, generator=generator).bfloat16()
        qo, ko = torch.randperm(11, generator=generator), torch.randperm(11, generator=generator)
        for orders in ((None, None), (qo, ko)):
            x = q.float() if orders[0] is None else q.float()[orders[0]]
            y = k.float() if orders[1] is None else k.float()[orders[1]]
            full = x @ y.T / (8 ** 0.5)
            expected = torch.stack([
                torch.stack([full[i:i+4, j:j+4].mean() for j in range(0, 11, 4)])
                for i in range(0, 11, 4)
            ])
            actual = pooled_scores(q, k, 4, *orders)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_original_frame_membership_undoes_permutation(self):
        q, k = torch.arange(48).reshape(12, 4).float(), torch.arange(48, 96).reshape(12, 4).float()
        order = torch.tensor([3, 8, 1, 6, 0, 11, 2, 4, 10, 5, 9, 7])
        restored = pooled_scores(q[order][order.argsort()], k[order][order.argsort()], 4)
        torch.testing.assert_close(restored, pooled_scores(q, k, 4))
        self.assertFalse(torch.equal(pooled_scores(q, k, 4, order, order), restored))

    def test_cluster_boundaries_include_empty_cluster_gaps(self):
        labels = torch.tensor([5, 0, 5, 2, 0, 2])
        torch.testing.assert_close(cluster_boundaries(labels, labels.argsort()), torch.tensor([0, 2, 4, 6]))

    def test_selection_preserves_random_states(self):
        with tempfile.TemporaryDirectory() as directory:
            python_state, torch_state, numpy_state = random.getstate(), torch.get_rng_state(), np.random.get_state()
            first = AttentionHeatmaps(directory, 30, seed=42)
            second = AttentionHeatmaps(directory, 30, seed=42)
            self.assertEqual(first.layers, second.layers)
            self.assertEqual(len(set(first.layers)), 7)
            self.assertEqual(python_state, random.getstate())
            torch.testing.assert_close(torch_state, torch.get_rng_state())
            self.assertEqual(numpy_state[0], np.random.get_state()[0])
            np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
            self.assertEqual(numpy_state[2:], np.random.get_state()[2:])

    def test_capture_files_branch_filter_and_same_inference_result(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = AttentionHeatmaps(directory, 1, branch="both")
            plain = WanNPUProcessor(0, sparse=True, qc=2, kc=3, fp_steps=0, fp_layers=0)
            traced = WanNPUProcessor(0, sparse=True, qc=2, kc=3, fp_steps=0,
                                     fp_layers=0, heatmaps=recorder)
            traced.token_grid = (2, 2, 2)
            q, k, v = [torch.randn(1, 2, 8, 128) for _ in range(3)]
            for branch in ("positive", "negative"):
                torch.manual_seed(51)
                expected = plain._self_attention(q, k, v)
                expected_state = torch.get_rng_state()
                torch.manual_seed(51)
                actual = traced._self_attention(q, k, v)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(torch.get_rng_state(), expected_state)
                target = Path(directory) / "step_001" / branch / "layer_00"
                self.assertEqual(len(list(target.glob("*.png"))), 2)
                with np.load(target / "head_00.npz") as artifact:
                    self.assertEqual(artifact["before"].shape, (2, 2))
                    expected_before = pooled_scores(q[0, 0], k[0, 0], 4).numpy()
                    np.testing.assert_allclose(artifact["before"], expected_before)
            recorder.finish()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(len(manifest["captured"]), 2)
            self.assertEqual(manifest["selected_layers"], [0])

    def test_requested_step_branch_and_grid_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = AttentionHeatmaps(directory, 1, branch="positive", step=2)
            q = torch.zeros(1, 1, 8, 4)
            labels = torch.zeros(1, 8, dtype=torch.long)
            recorder.capture(0, 1, "positive", q, q, labels, labels, (2, 2, 2))
            recorder.capture(0, 2, "negative", q, q, labels, labels, (2, 2, 2))
            self.assertEqual(recorder.done, set())
            self.assertEqual(list(Path(directory).rglob("*.png")), [])
            with self.assertRaisesRegex(ValueError, "Token grid"):
                recorder.capture(0, 2, "positive", q, q, labels, labels, (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
