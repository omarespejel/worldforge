import tempfile
import unittest

from _helpers import require_installed_module


class WorldForgePythonPackageSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worldforge = require_installed_module("worldforge")

    def test_world_prediction_compare_and_persistence_flow(self) -> None:
        worldforge = self.worldforge
        with tempfile.TemporaryDirectory(prefix="worldforge-python-smoke-") as state_dir:
            forge = worldforge.WorldForge(state_dir=state_dir)
            self.assertIn("mock", forge.providers())

            world = forge.create_world("kitchen-counter", "mock")
            world.add_object(
                worldforge.SceneObject(
                    "red_mug",
                    worldforge.Position(0.0, 0.8, 0.0),
                    worldforge.BBox(
                        worldforge.Position(-0.05, 0.75, -0.05),
                        worldforge.Position(0.05, 0.85, 0.05),
                    ),
                )
            )

            prediction = world.predict(worldforge.Action.move_to(0.25, 0.8, 0.0, 1.0), steps=4)
            self.assertEqual(prediction.provider, "mock")
            self.assertGreaterEqual(prediction.confidence, 0.0)

            comparison = world.compare(
                worldforge.Action.move_to(0.3, 0.8, 0.0, 1.0),
                ["mock"],
                steps=2,
            )
            self.assertEqual(comparison.best_prediction().provider, "mock")

            world_id = forge.save_world(world)
            self.assertIn(world_id, forge.list_worlds())

            loaded = forge.load_world(world_id)
            self.assertIn("red_mug", loaded.list_objects())

    def test_generation_transfer_eval_and_verification_helpers(self) -> None:
        worldforge = self.worldforge
        forge = worldforge.WorldForge()

        clip = forge.generate("A cube rolling across a table", "mock", duration_seconds=1.0)
        self.assertGreaterEqual(clip.frame_count, 1)

        transferred = forge.transfer(clip, "mock", width=320, height=180, fps=12.0)
        self.assertEqual(transferred.resolution, (320, 180))

        suites = worldforge.list_eval_suites()
        self.assertIn("physics", suites)

        results = worldforge.run_eval("physics", "mock")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].provider, "mock")

        proof = worldforge.prove_inference(b"model", b"input", b"output")
        valid, details = proof.verify()
        self.assertTrue(valid)
        self.assertTrue(details)

    def test_typed_verification_bundle_flow(self) -> None:
        worldforge = self.worldforge
        from worldforge.verify import ZkVerifier, prove_inference_transition_bundle

        world = worldforge.World("verify-python", "mock")
        world.add_object(
            worldforge.SceneObject(
                "cube",
                worldforge.Position(0.0, 0.5, 0.0),
                worldforge.BBox(
                    worldforge.Position(-0.05, 0.45, -0.05),
                    worldforge.Position(0.05, 0.55, 0.05),
                ),
            )
        )

        before = world.to_json()
        prediction = world.predict(worldforge.Action.move_to(0.2, 0.5, 0.0, 1.0), steps=2)
        self.assertEqual(prediction.provider, "mock")
        world.predict(worldforge.Action.move_to(0.3, 0.5, 0.0, 1.0), steps=2)

        latest_bundle = world.prove_latest_inference_bundle()
        self.assertEqual(latest_bundle.provider, "mock")
        latest_report = latest_bundle.verify()
        self.assertTrue(latest_report.current_verification.valid)
        self.assertTrue(latest_report.verification_matches_recorded)

        detached_bundle = prove_inference_transition_bundle(before, world.to_json(), provider="mock")
        self.assertEqual(detached_bundle.provider, "mock")

        plan = world.plan(goal="spawn cube", max_steps=4, provider="mock")
        guardrail_bundle = plan.prove_guardrail_bundle()
        self.assertGreaterEqual(guardrail_bundle.action_count, 1)

        verifier = ZkVerifier()
        guardrail_report = verifier.verify_guardrail_bundle(guardrail_bundle)
        self.assertTrue(guardrail_report.current_verification.valid)

        provenance_bundle = world.prove_provenance_bundle(
            source_label="python-smoke",
            timestamp=1710000000,
        )
        self.assertEqual(provenance_bundle.timestamp, 1710000000)
        provenance_report = verifier.verify_provenance_bundle(provenance_bundle)
        self.assertTrue(provenance_report.current_verification.valid)


if __name__ == "__main__":
    unittest.main()
