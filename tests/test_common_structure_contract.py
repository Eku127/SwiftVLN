from __future__ import annotations

import unittest
from pathlib import Path

import swiftvln.common as common


REPO_ROOT = Path(__file__).resolve().parents[1]


class CommonStructureContractTest(unittest.TestCase):
    def test_common_root_is_not_a_cross_package_facade(self):
        self.assertEqual(common.__all__, ())
        self.assertFalse(hasattr(common, "EnvWrapper"))
        self.assertFalse(hasattr(common, "HistoryTokenCompressor"))

    def test_internal_code_uses_concrete_common_modules(self):
        internal_imports = []
        for path in (REPO_ROOT / "src/swiftvln").rglob("*.py"):
            if path == REPO_ROOT / "src/swiftvln/common/__init__.py":
                continue
            source = path.read_text(encoding="utf-8")
            if "from swiftvln.common import" in source:
                internal_imports.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(internal_imports, [])


if __name__ == "__main__":
    unittest.main()
