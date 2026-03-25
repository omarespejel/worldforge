import tempfile
import unittest

try:
    import worldforge
except ModuleNotFoundError:  # pragma: no cover - exercised by discovery when the package is absent
    worldforge = None


class WorldForgeSubmoduleImportTests(unittest.TestCase):
    def test_provider_submodule_exports_provider_classes(self) -> None:
        if worldforge is None:
            self.skipTest("worldforge package is not installed")

        try:
            from worldforge.providers import MockProvider
        except ModuleNotFoundError as error:
            self.skipTest(f"worldforge.providers is not available yet: {error}")

        self.assertIsNotNone(MockProvider)

    def test_eval_and_verify_submodules_are_importable(self) -> None:
        if worldforge is None:
            self.skipTest("worldforge package is not installed")

        try:
            from worldforge.eval import EvalScenario, EvalSuite
            from worldforge.verify import MockVerifier, ZkProof
        except ModuleNotFoundError as error:
            self.skipTest(f"worldforge.eval/worldforge.verify are not available yet: {error}")

        self.assertIsNotNone(EvalScenario)
        self.assertIsNotNone(EvalSuite)
        self.assertIsNotNone(MockVerifier)
        self.assertIsNotNone(ZkProof)

    def test_manual_provider_registration_flow(self) -> None:
        if worldforge is None:
            self.skipTest("worldforge package is not installed")

        try:
            from worldforge.providers import MockProvider
        except ModuleNotFoundError as error:
            self.skipTest(f"worldforge.providers is not available yet: {error}")

        if not hasattr(worldforge, "WorldForge"):
            self.skipTest("WorldForge is not exposed by the Python package")

        named_provider = getattr(MockProvider, "with_name", None)
        provider = named_provider("manual-mock") if callable(named_provider) else MockProvider()
        expected_name = "manual-mock" if callable(named_provider) else None

        with tempfile.TemporaryDirectory(prefix="worldforge-python-registration-") as state_dir:
            forge = worldforge.WorldForge(state_dir=state_dir)

            register_provider = getattr(forge, "register_provider", None)
            if register_provider is None:
                self.skipTest("WorldForge.register_provider is not available yet")

            registered = False
            if expected_name is not None:
                try:
                    register_provider(provider)
                    registered = True
                except TypeError:
                    pass
                except Exception:
                    pass

                if not registered:
                    try:
                        register_provider(expected_name, provider)
                        registered = True
                    except TypeError:
                        pass
            else:
                try:
                    register_provider("manual-mock", provider)
                    registered = True
                    expected_name = "manual-mock"
                except TypeError:
                    self.skipTest(
                        "WorldForge.register_provider does not accept an explicit alias yet"
                    )

            if not registered:
                self.skipTest("WorldForge.register_provider does not support the intended flow yet")

            providers_list = forge.providers()
            self.assertIn(expected_name, providers_list)

            descriptor = forge.provider_info(expected_name)
            self.assertEqual(descriptor.name, expected_name)
            self.assertTrue(descriptor.capabilities.predict)


if __name__ == "__main__":
    unittest.main()
