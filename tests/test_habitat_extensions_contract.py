import unittest


class HabitatExtensionsContractTest(unittest.TestCase):
    def test_only_required_custom_measures_remain(self):
        from swiftvln.habitat_extensions import measures

        self.assertEqual(
            {
                name
                for name in vars(measures)
                if isinstance(vars(measures)[name], type)
                and vars(measures)[name].__module__ == measures.__name__
            },
            {"OracleNavigationError", "OracleSuccess"},
        )
        self.assertEqual(
            measures.OracleNavigationError.cls_uuid,
            "oracle_navigation_error",
        )
        self.assertEqual(measures.OracleSuccess.cls_uuid, "oracle_success")


if __name__ == "__main__":
    unittest.main()
