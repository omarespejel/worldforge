import unittest

from _helpers import require_installed_module


class WorldForgeInstallContractTests(unittest.TestCase):
    def test_worldforge_package_and_submodules_import(self) -> None:
        worldforge = require_installed_module("worldforge")

        from worldforge.eval import EvalSuite
        from worldforge.providers import MockProvider
        from worldforge.verify import ZkVerifier

        self.assertIsNotNone(worldforge)
        self.assertTrue(hasattr(worldforge, "WorldForge"))
        self.assertIsNotNone(EvalSuite)
        self.assertIsNotNone(MockProvider)
        self.assertIsNotNone(ZkVerifier)


if __name__ == "__main__":
    unittest.main()
