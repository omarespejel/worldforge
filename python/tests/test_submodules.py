import tempfile
import unittest

try:
    import worldforge
except ModuleNotFoundError:  # pragma: no cover - exercised by discovery when the package is absent
    worldforge = None


class WorldForgeSubmoduleImportTests(unittest.TestCase):
    @unittest.skipUnless(worldforge is not None, "worldforge package is not installed")
    def test_provider_submodule_exports_provider_classes(self) -> None:
        from worldforge.providers import (
            CosmosProvider,
            GenieProvider,
            JepaProvider,
            MockProvider,
            RunwayProvider,
        )

        self.assertIsNotNone(CosmosProvider)
        self.assertIsNotNone(GenieProvider)
        self.assertIsNotNone(JepaProvider)
        self.assertIsNotNone(MockProvider)
        self.assertIsNotNone(RunwayProvider)

    @unittest.skipUnless(worldforge is not None, "worldforge package is not installed")
    def test_eval_and_verify_submodules_are_importable(self) -> None:
        from worldforge.eval import EvalScenario, EvalSuite, PhysicsEval
        from worldforge.verify import MockVerifier, ZkProof, ZkVerifier

        self.assertIsNotNone(EvalScenario)
        self.assertIsNotNone(EvalSuite)
        self.assertIsNotNone(PhysicsEval)
        self.assertIsNotNone(MockVerifier)
        self.assertIsNotNone(ZkProof)
        self.assertIsNotNone(ZkVerifier)

    @unittest.skipUnless(worldforge is not None, "worldforge package is not installed")
    def test_manual_provider_registration_flow(self) -> None:
        from worldforge.providers import MockProvider

        with tempfile.TemporaryDirectory(prefix="worldforge-python-registration-") as state_dir:
            forge = worldforge.WorldForge(state_dir=state_dir)
            provider = MockProvider(name="manual-mock")
            forge.register_provider(provider)
            providers_list = forge.providers()
            self.assertIn("manual-mock", providers_list)

            descriptor = forge.provider_info("manual-mock")
            self.assertEqual(descriptor.name, "manual-mock")
            self.assertTrue(descriptor.capabilities.predict)

            world = forge.create_world("manual-world", "manual-mock")
            prediction = world.predict(worldforge.Action.move_to(0.2, 0.8, 0.0), steps=2)
            self.assertEqual(prediction.provider, "manual-mock")


if __name__ == "__main__":
    unittest.main()
