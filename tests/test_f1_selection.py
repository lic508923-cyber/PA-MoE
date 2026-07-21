import unittest

from pa_moelog.utils import compute_binary_metrics, select_best_f1_threshold


class F1ThresholdSelectionTest(unittest.TestCase):
    def test_exact_threshold_beats_coarse_half_step(self):
        labels = [1, 0, 1, 0]
        scores = [0.491, 0.490, 0.489, 0.100]
        selected = select_best_f1_threshold(labels, scores)
        self.assertAlmostEqual(selected["threshold"], 0.489)
        self.assertAlmostEqual(selected["f1"], 0.8)
        self.assertGreater(selected["f1"], compute_binary_metrics(labels, scores, 0.5)["f1"])

    def test_tied_scores_enter_together(self):
        selected = select_best_f1_threshold([1, 0, 1], [0.8, 0.8, 0.2])
        self.assertEqual(selected["threshold"], 0.2)
        self.assertAlmostEqual(selected["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
