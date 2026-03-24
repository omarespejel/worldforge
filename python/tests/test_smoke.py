import tempfile
import unittest

import worldforge


class WorldForgePythonPackageSmokeTests(unittest.TestCase):
    def test_world_prediction_compare_and_persistence_flow(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
