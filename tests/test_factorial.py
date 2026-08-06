import unittest
from geoai_gid_yrb.factorial import factorial_effects, summarize_scene_cells


class FactorialTests(unittest.TestCase):
    def test_effects(self):
        result = factorial_effects({"P1": 0.90, "P2": 0.70, "P3": 0.80, "P4": 0.95})
        self.assertAlmostEqual(
            result["main_effect_resolution_native_minus_coarse"], 0.175
        )
        self.assertAlmostEqual(
            result["main_effect_labels_fine_minus_coarse"], -0.075
        )
        self.assertAlmostEqual(result["interaction"], 0.05)
        self.assertEqual(result["dominant_absolute_main_effect"], "resolution")

    def test_scene_summary(self):
        result = summarize_scene_cells({
            "a": {"P1": 0.8, "P2": 0.6, "P3": 0.7, "P4": 0.9},
            "b": {"P1": 1.0, "P2": 0.8, "P3": 0.9, "P4": 1.0},
        })
        self.assertEqual(result["cells"]["P1"]["n"], 2)
        self.assertAlmostEqual(result["cells"]["P1"]["mean"], 0.9)


if __name__ == "__main__":
    unittest.main()
