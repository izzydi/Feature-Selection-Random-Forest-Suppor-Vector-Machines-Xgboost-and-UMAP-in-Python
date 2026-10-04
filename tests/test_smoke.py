import unittest

import numpy as np
import pandas as pd

from src.feature_selection_pipeline import EXPECTED_COLUMNS, score, split_xy


class FeatureSelectionSmokeTests(unittest.TestCase):
    def test_split_xy_uses_last_expected_column_as_target(self):
        df = pd.DataFrame(np.arange(4 * EXPECTED_COLUMNS).reshape(4, EXPECTED_COLUMNS))
        x, y = split_xy(df)
        self.assertEqual(x.shape, (4, EXPECTED_COLUMNS - 1))
        self.assertEqual(y.shape, (4,))

    def test_score_returns_perfect_metrics_for_perfect_predictions(self):
        y = np.array([0, 1, 0, 1])
        metrics = score(y, y.copy())
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["f1_weighted"], 1.0)


if __name__ == "__main__":
    unittest.main()
