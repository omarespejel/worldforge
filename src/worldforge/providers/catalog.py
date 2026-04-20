"""Provider catalog and registration policy for WorldForge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from worldforge.models import ProviderEvent

from .base import BaseProvider
from .cosmos import CosmosProvider
from .gr00t import GrootPolicyClientProvider
from .lerobot import LeRobotPolicyProvider
from .leworldmodel import LeWorldModelProvider
from .mock import MockProvider
from .remote import GenieProvider, JepaProvider
from .runway import RunwayProvider

ProviderEventHandler = Callable[[ProviderEvent], None] | None
ProviderFactory = Callable[[ProviderEventHandler], BaseProvider]


@dataclass(frozen=True, slots=True)
class ProviderCatalogEntry:
    """Factory and registration policy for an in-repo provider adapter."""

    name: str
    factory: ProviderFactory
    always_register: bool = False

    def create(self, *, event_handler: ProviderEventHandler = None) -> BaseProvider:
        return self.factory(event_handler)


def _mock(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return MockProvider(event_handler=event_handler)


def _cosmos(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return CosmosProvider(event_handler=event_handler)


def _runway(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return RunwayProvider(event_handler=event_handler)


def _leworldmodel(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return LeWorldModelProvider(event_handler=event_handler)


def _gr00t(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return GrootPolicyClientProvider(event_handler=event_handler)


def _lerobot(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return LeRobotPolicyProvider(event_handler=event_handler)


def _jepa(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return JepaProvider(event_handler=event_handler)


def _genie(event_handler: ProviderEventHandler = None) -> BaseProvider:
    return GenieProvider(event_handler=event_handler)


PROVIDER_CATALOG: tuple[ProviderCatalogEntry, ...] = (
    ProviderCatalogEntry("mock", _mock, always_register=True),
    ProviderCatalogEntry("cosmos", _cosmos),
    ProviderCatalogEntry("runway", _runway),
    ProviderCatalogEntry("leworldmodel", _leworldmodel),
    ProviderCatalogEntry("gr00t", _gr00t),
    ProviderCatalogEntry("lerobot", _lerobot),
    ProviderCatalogEntry("jepa", _jepa),
    ProviderCatalogEntry("genie", _genie),
)


def create_known_providers(
    *, event_handler: ProviderEventHandler = None
) -> tuple[BaseProvider, ...]:
    """Instantiate every in-repo provider adapter without registering it."""

    return tuple(entry.create(event_handler=event_handler) for entry in PROVIDER_CATALOG)
