from __future__ import annotations

import unittest

from experiments.phase_g_v2_2.baselines import calculate_exact_baselines


class BayesBaselineTests(unittest.TestCase):
    def test_all_six_exact_joint_baselines(self):
        baselines = calculate_exact_baselines()

        expected = {
            "balanced": (0.85, 0.30265306951792803),
            "image_dominant": (0.95, 0.22799857626959324),
            "clinical_dominant": (0.95, 0.22799857626959324),
            "redundant": (0.85, 0.6641247746711061),
            "conflicting": (0.675, 0.5119965801280856),
            "synergistic": (0.727, 1.0255964798242845),
        }
        for name, (accuracy, nll) in expected.items():
            self.assertAlmostEqual(baselines[name]["accuracy"], accuracy, places=6)
            self.assertAlmostEqual(baselines[name]["nll"], nll, places=6)


if __name__ == "__main__":
    unittest.main()
