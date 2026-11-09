from __future__ import annotations

import unittest

from experiments.halora.schedule import (
    should_apply_regularizer,
    should_refresh_noise,
)


class ScheduleTest(unittest.TestCase):
    def test_1300_step_schedule(self) -> None:
        refreshes = [
            step for step in range(1300) if should_refresh_noise(step, 400)
        ]
        regularized = [
            step for step in range(1300) if should_apply_regularizer(step, 400)
        ]
        self.assertEqual(refreshes, [0, 400, 800, 1200])
        self.assertEqual(regularized, refreshes)


if __name__ == "__main__":
    unittest.main()
