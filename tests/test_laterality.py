import unittest

import torch

from laterality import BRAIN_LEFT_RIGHT_PAIRS, correct_brain_laterality_scores


class BrainLateralityTest(unittest.TestCase):
    def test_all_bilateral_brain_regions_are_paired(self):
        self.assertEqual(len(BRAIN_LEFT_RIGHT_PAIRS), 40)
        labels = [label for pair in BRAIN_LEFT_RIGHT_PAIRS for label in pair]
        self.assertEqual(len(labels), len(set(labels)))

    def test_score_channels_are_swapped(self):
        scores = torch.arange(215, dtype=torch.float32).reshape(1, 215, 1, 1, 1)
        corrected = correct_brain_laterality_scores(scores)
        for left_label, right_label in BRAIN_LEFT_RIGHT_PAIRS:
            self.assertEqual(corrected[0, left_label].item(), float(right_label))
            self.assertEqual(corrected[0, right_label].item(), float(left_label))
        self.assertEqual(corrected[0, 15].item(), 15.0)
        self.assertEqual(corrected[0, 16].item(), 16.0)


if __name__ == "__main__":
    unittest.main()
