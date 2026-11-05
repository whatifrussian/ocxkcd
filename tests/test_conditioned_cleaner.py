import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocxkcd._conditioned_cleaner import ConditionedCleaner, apply_coverage, bounded_mask, input_array, predict, predict_detections


class CleanerTests(unittest.TestCase):
    def test_packaged_model_loads_and_rejects_wrong_architecture(self):
        import tempfile
        from pathlib import Path

        from ocxkcd._conditioned_cleaner import load_checkpoint
        from ocxkcd.cli import bundled_checkpoint

        path = bundled_checkpoint("conditioned-cleaner-v1.pt")
        model, checkpoint = load_checkpoint(path)
        self.assertFalse(model.training)
        self.assertEqual(checkpoint["threshold"], 0.3)
        self.assertNotIn("trainingArgs", checkpoint)
        with tempfile.TemporaryDirectory() as folder:
            invalid = Path(folder) / "invalid.pt"
            torch.save({"architecture": "wrong"}, invalid)
            with self.assertRaisesRegex(ValueError, "Not a conditioned"):
                load_checkpoint(invalid)

    def test_input_keeps_original_alpha_separate_from_visible_rgb(self):
        rgba = np.array([[[0, 0, 0, 0], [0, 0, 0, 128], [0, 0, 0, 255]]], np.uint8)
        x = input_array(rgba, np.zeros((1, 3), np.uint8))
        np.testing.assert_allclose(x[0, :, 3], [0, 128 / 255, 1])
        np.testing.assert_allclose(x[0, :, 0], [1, 127 / 255, 0], atol=1e-6)

    def test_no_proposal_cannot_erase_anything(self):
        self.assertFalse(bounded_mask(np.ones((40, 50)), np.zeros((40, 50), np.uint8), 0.5).any())

    def test_bounding_does_not_reach_remote_artwork(self):
        proposal = np.zeros((40, 50), np.uint8)
        proposal[20, 25] = 255
        result = bounded_mask(np.ones((40, 50)), proposal, 0.5, radius=2)
        self.assertEqual(int(result.sum()), 25)
        self.assertFalse(result[0, 0])

    def test_tiled_prediction_covers_edges_without_seams(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], 1, *x.shape[-2:]), device=x.device)

        rgba = np.zeros((301, 397, 4), np.uint8)
        output = predict(Constant(), rgba, np.zeros((301, 397), np.uint8), "cpu")
        self.assertEqual(output.shape, (301, 397))
        np.testing.assert_allclose(output, 0.5, atol=1e-6)

    def test_model_accepts_five_channels_and_odd_dimensions(self):
        torch.set_num_threads(2)
        model = ConditionedCleaner().eval()
        with torch.inference_mode():
            result = model(torch.zeros(1, 5, 65, 79))
        self.assertEqual(tuple(result.shape), (1, 1, 65, 79))

    def test_background_transfer_preserves_color_alpha_and_artwork(self):
        source = np.full((7, 9, 4), 255, np.uint8)
        source[0, 0] = [20, 30, 40, 255]
        clean = source.copy()
        clean[3, 2] = [50, 70, 90, 80]
        clean[3, 6] = [0, 0, 0, 255]
        proposal = np.zeros((7, 9), np.uint8)
        proposal[3, 2] = 255
        proposal[3, 6] = 255
        selected = np.zeros((7, 9), bool)
        selected[3, 1:3] = True
        selected[3, 6:8] = True
        result = apply_coverage(source, clean, proposal, selected)
        np.testing.assert_array_equal(result[3, 1], [50, 70, 90, 80])
        np.testing.assert_array_equal(result[3, 7], [0, 0, 0, 255])
        np.testing.assert_array_equal(result[~selected], source[~selected])

    def test_missing_detections_preserve_baseline_mask(self):
        proposal = np.zeros((32, 40), np.uint8)
        proposal[10:15, 12:18] = 255
        output = predict_detections(None, np.zeros((32, 40, 4), np.uint8), proposal, [], "cpu")
        np.testing.assert_array_equal(output > 0, proposal > 0)

    def test_overlapping_rotated_crops_blend_without_nan(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], 1, *x.shape[-2:]), device=x.device)

        boxes = [
            {"box": {"x": 0, "y": 2, "width": 30, "height": 10, "angle": 25}},
            {"box": {"x": 20, "y": 20, "width": 20, "height": 10, "angle": -15}},
        ]
        output = predict_detections(Constant(), np.zeros((40, 50, 4), np.uint8), np.zeros((40, 50), np.uint8), boxes, "cpu")
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output, 0.5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
