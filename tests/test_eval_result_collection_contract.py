from __future__ import annotations

import unittest

from swiftvln.scripts.eval.collect_eval_results import (
    infer_plan,
    normalize_overlap_setting_key,
    overlap_variant_rank,
)


PREFIX = "swiftvln-satnav-3b-1ep-f32s4-overlap0-"


class EvalResultCollectionContractTest(unittest.TestCase):
    def test_plan_uses_experiment_spec(self):
        cases = {
            f"{PREFIX}pf-h8-pool-s2-noembed": "baseline",
            f"{PREFIX}pf-h8-random-pool-s2-noembed": "baseline + random",
            f"{PREFIX}gtc-k512-noembed": "baseline + gtc-k512",
            f"{PREFIX}sgtc-k512-noembed": "baseline + sgtc-k512",
            f"{PREFIX}map-g1000-l400-r448-d20-s2-noembed": "baseline + map",
            f"{PREFIX}pf-h8-pool-s2-initial-noembed": "baseline + initial",
        }
        for model_name, expected in cases.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(infer_plan(model_name, "swiftvln"), expected)

    def test_variants_share_a_setting_key_and_have_stable_order(self):
        baseline = f"{PREFIX}pf-h8-pool-s2-noembed"
        map_name = f"{PREFIX}map-g1000-l400-r448-d20-s2-noembed"
        gtc = f"{PREFIX}gtc-k512-noembed"
        sgtc = f"{PREFIX}sgtc-k512-noembed"
        self.assertEqual(
            {normalize_overlap_setting_key(name) for name in (baseline, map_name, gtc, sgtc)},
            {normalize_overlap_setting_key(baseline)},
        )
        self.assertEqual(
            [overlap_variant_rank(name) for name in (baseline, map_name, gtc, sgtc)],
            [10, 50, 60, 70],
        )


if __name__ == "__main__":
    unittest.main()
