import unittest
from typing import List

import gin

from src.utils import get_single_experiment_results


class TestHypothesis(unittest.TestCase):
    def setUp(self) -> None:
        gin.parse_config("configs/config.gin")

    @staticmethod
    def _get_single_exp_results(experts_features_importance: List[float], seed: int) -> float:
        loss, acc = get_single_experiment_results(batch_size=1000,
                                                  epochs=100,
                                                  tensorboard_logs="/data/logs/test_logs",
                                                  seed=seed)
        return loss

    def test_hyp(self):
        correct_importance_variations = [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.5, 0.0, 0.0],
            [0.8, 0.8, 0.3, 0.05, 0.05]
        ]
        incorrect_importance_variations = [
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.5, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.5, 1.0, 0.0],
            [0.0, 0.0, 0.5, 1.0, 1.0]
        ]

        for seed_val in range(0, 5):
            with self.subTest(f"seed value = {seed_val}"):
                correct_values_losses = [
                    self._get_single_exp_results(exp_import, seed_val) for exp_import in correct_importance_variations
                ]

                incorrect_values_losses = [
                    self._get_single_exp_results(exp_import, seed_val) for exp_import in incorrect_importance_variations
                ]

                max_correct_loss = max(correct_values_losses)
                min_incorrect_loss = min(incorrect_values_losses)

                self.assertLessEqual(max_correct_loss, min_incorrect_loss)
