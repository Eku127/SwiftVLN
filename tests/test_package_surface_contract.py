from __future__ import annotations

import unittest

import swiftvln.s2r as s2r
import swiftvln.scripts.data_process as data_process


class PackageSurfaceContractTest(unittest.TestCase):
    def test_s2r_root_requires_explicit_submodule_imports(self):
        self.assertEqual(s2r.__all__, ())
        self.assertFalse(hasattr(s2r, "__getattr__"))
        self.assertFalse(hasattr(s2r, "SatDronePairDataset"))

    def test_data_process_root_has_no_import_side_effects(self):
        self.assertEqual(data_process.__all__, ())
        self.assertFalse(hasattr(data_process, "process_episodes"))
        self.assertFalse(hasattr(data_process, "run_all"))


if __name__ == "__main__":
    unittest.main()
