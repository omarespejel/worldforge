import json
import tempfile
import unittest

from _helpers import require_installed_module


class WorldForgePythonParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worldforge = require_installed_module("worldforge")

    def test_prompt_seeded_world_health_reason_embed_and_history_flow(self) -> None:
        worldforge = self.worldforge

        with tempfile.TemporaryDirectory(prefix="worldforge-python-parity-") as state_dir:
            forge = worldforge.WorldForge(state_dir=state_dir)

            seeded = forge.create_world_from_prompt(
                "A kitchen with a mug",
                provider="mock",
                name="seeded-kitchen",
            )

            self.assertEqual(seeded.name, "seeded-kitchen")
            self.assertEqual(seeded.description, "A kitchen with a mug")
            self.assertGreaterEqual(seeded.object_count, 2)

            mock_health = forge.provider_health("mock")
            self.assertEqual(mock_health.name, "mock")
            self.assertTrue(mock_health.healthy)

            predict_healths = {
                health.name: health for health in forge.provider_healths("predict")
            }
            self.assertIn("mock", predict_healths)
            self.assertTrue(predict_healths["mock"].healthy)

            prediction = seeded.predict(
                worldforge.Action.move_to(0.25, 0.5, 0.0, 1.0),
                steps=2,
            )
            self.assertEqual(prediction.provider, "mock")
            self.assertGreaterEqual(seeded.history_length, 2)

            history = seeded.history()
            self.assertIsNone(history[0].action_json)
            self.assertIsNotNone(history[-1].action_json)

            checkpoint = seeded.history_state(0)
            self.assertEqual(checkpoint.step, 0)
            self.assertEqual(checkpoint.history_length, 1)

            seeded.restore_history(0)
            self.assertEqual(seeded.step, 0)

            saved_id = forge.save_world(seeded)
            forked = forge.fork_world(
                saved_id,
                history_index=0,
                name="seeded-kitchen-branch",
            )
            self.assertNotEqual(forked.id, saved_id)
            self.assertEqual(forked.name, "seeded-kitchen-branch")
            self.assertEqual(forked.history_length, 1)

            reasoning = forge.reason(
                "mock",
                "how many objects are here?",
                world=seeded,
            )
            self.assertTrue(reasoning.answer)
            self.assertGreaterEqual(reasoning.confidence, 0.0)
            self.assertGreaterEqual(len(reasoning.evidence), 1)

            embedding = forge.embed("mock", text="a mug on a kitchen counter")
            self.assertEqual(embedding.provider, "mock")
            self.assertEqual(embedding.model, "mock-embedding-v1")
            self.assertEqual(embedding.shape, [32])
            self.assertEqual(len(embedding.vector), 32)

    def test_structured_goal_json_planning_and_comparison_artifacts(self) -> None:
        worldforge = self.worldforge

        with tempfile.TemporaryDirectory(prefix="worldforge-python-plan-") as state_dir:
            forge = worldforge.WorldForge(state_dir=state_dir)

            world = forge.create_world("goal-json-world", "mock")
            position = worldforge.Position(0.0, 0.5, 0.0)
            bbox = worldforge.BBox(
                worldforge.Position(-0.1, 0.4, -0.1),
                worldforge.Position(0.1, 0.6, 0.1),
            )
            ball = worldforge.SceneObject("ball", position, bbox)
            ball_id = ball.id
            world.add_object(ball)

            goal_json = json.dumps(
                {
                    "type": "condition",
                    "condition": {
                        "ObjectAt": {
                            "object": ball_id,
                            "position": {"x": 1.0, "y": 0.5, "z": 0.0},
                            "tolerance": 0.05,
                        }
                    },
                }
            )

            plan = world.plan(
                goal_json=goal_json,
                max_steps=4,
                timeout_seconds=10.0,
                provider="mock",
                planner="sampling",
                num_samples=48,
                top_k=5,
            )
            self.assertGreater(plan.action_count, 0)

            plan_json = json.loads(plan.to_json())
            final_state = plan_json["predicted_states"][-1]
            final_ball = next(
                obj
                for obj in final_state["scene"]["objects"].values()
                if obj["name"] == "ball"
            )
            self.assertAlmostEqual(
                final_ball["pose"]["position"]["x"], 1.0, delta=0.15
            )
            self.assertAlmostEqual(
                final_ball["pose"]["position"]["y"], 0.5, delta=0.15
            )

            comparison_world = forge.create_world("comparison-world", "mock")
            comparison_world.add_object(
                worldforge.SceneObject(
                    "cube",
                    worldforge.Position(0.0, 0.8, 0.0),
                    worldforge.BBox(
                        worldforge.Position(-0.05, 0.75, -0.05),
                        worldforge.Position(0.05, 0.85, 0.05),
                    ),
                )
            )

            first_prediction = comparison_world.predict(
                worldforge.Action.move_to(0.25, 0.8, 0.0, 1.0),
                steps=2,
            )
            second_prediction = comparison_world.predict(
                worldforge.Action.move_to(0.35, 0.8, 0.0, 1.0),
                steps=2,
            )

            comparison = forge.compare([first_prediction, second_prediction])
            self.assertEqual(comparison.prediction_count, 2)
            self.assertEqual(comparison.best_prediction().provider, "mock")

            artifacts = comparison.artifacts()
            self.assertEqual(set(artifacts), {"json", "markdown", "csv"})
            self.assertTrue(comparison.to_markdown().startswith("#"))
            self.assertIn("provider", comparison.to_csv())

    def test_eval_and_verify_submodules_render_reports(self) -> None:
        worldforge = self.worldforge

        from worldforge.eval import EvalSuite
        from worldforge.verify import MockVerifier, prove_inference_transition_bundle

        with tempfile.TemporaryDirectory(prefix="worldforge-python-eval-") as state_dir:
            forge = worldforge.WorldForge(state_dir=state_dir)
            world = forge.create_world("eval-world", "mock")
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

            suite = EvalSuite.from_builtin("physics")
            report = suite.run_report_data("mock", world=world, forge=forge)
            self.assertIn("Physics", report.suite)
            self.assertGreaterEqual(len(report.provider_summaries), 1)
            self.assertEqual(report.provider_summaries[0].provider, "mock")

            report_artifacts = suite.run_report_artifacts(
                providers="mock",
                world=world,
                forge=forge,
            )
            self.assertEqual(set(report_artifacts), {"json", "markdown", "csv"})
            self.assertTrue(report_artifacts["markdown"].startswith("# Evaluation Report"))

            before = world.to_json()
            prediction = world.predict(
                worldforge.Action.move_to(0.2, 0.5, 0.0, 1.0),
                steps=2,
            )
            self.assertEqual(prediction.provider, "mock")

            archived_bundle = prediction.prove_inference_bundle("mock")
            archived_report = archived_bundle.verify()
            self.assertTrue(archived_report.current_verification.valid)
            self.assertTrue(archived_report.verification_matches_recorded)

            transition_bundle = prove_inference_transition_bundle(
                before,
                world.to_json(),
                provider="mock",
            )
            self.assertEqual(transition_bundle.provider, "mock")

            verifier = MockVerifier()
            transition_report = verifier.verify_inference_bundle(transition_bundle)
            self.assertTrue(transition_report.current_verification.valid)


if __name__ == "__main__":
    unittest.main()
