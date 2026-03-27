import tempfile
import unittest

from _helpers import require_installed_module


class WorldForgeSubmoduleImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worldforge = require_installed_module("worldforge")

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

    def test_eval_and_verify_submodules_are_importable(self) -> None:
        from worldforge.eval import EvalScenario, EvalSuite, PhysicsEval
        from worldforge.verify import (
            GuardrailBundle,
            GuardrailVerificationReport,
            InferenceBundle,
            InferenceVerificationReport,
            MockVerifier,
            ProvenanceBundle,
            ProvenanceVerificationReport,
            VerificationResult,
            ZkProof,
            ZkVerifier,
        )

        self.assertIsNotNone(EvalScenario)
        self.assertIsNotNone(EvalSuite)
        self.assertIsNotNone(PhysicsEval)
        self.assertIsNotNone(GuardrailBundle)
        self.assertIsNotNone(GuardrailVerificationReport)
        self.assertIsNotNone(InferenceBundle)
        self.assertIsNotNone(InferenceVerificationReport)
        self.assertIsNotNone(MockVerifier)
        self.assertIsNotNone(ProvenanceBundle)
        self.assertIsNotNone(ProvenanceVerificationReport)
        self.assertIsNotNone(VerificationResult)
        self.assertIsNotNone(ZkProof)
        self.assertIsNotNone(ZkVerifier)

    def test_manual_provider_registration_flow(self) -> None:
        from worldforge.providers import MockProvider

        worldforge = self.worldforge
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

            plan = world.plan(goal="spawn cube", max_steps=3, verify_backend="mock")
            self.assertIsNotNone(plan.verification_proof)
            self.assertEqual(plan.verification_proof.backend, "Mock")


if __name__ == "__main__":
    unittest.main()
