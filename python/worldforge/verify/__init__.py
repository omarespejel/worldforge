"""Verification helpers for the pure-Python WorldForge runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge._core import JsonDict, json_dumps, json_hash


@dataclass(slots=True)
class VerificationResult:
    """Result of checking a proof or verification bundle."""

    valid: bool
    details: str

    def to_dict(self) -> JsonDict:
        return {"valid": self.valid, "details": self.details}


@dataclass(slots=True)
class ZkProof:
    """Portable proof placeholder used during the Python-first migration."""

    backend: str
    claim_type: str
    payload_hash: str

    def verify(self) -> tuple[bool, str]:
        return True, f"{self.claim_type}:{self.payload_hash}"

    def to_dict(self) -> JsonDict:
        return {
            "backend": self.backend,
            "claim_type": self.claim_type,
            "payload_hash": self.payload_hash,
        }


@dataclass(slots=True)
class InferenceVerificationReport:
    """Verification report for an inference bundle."""

    current_verification: VerificationResult
    verification_matches_recorded: bool


@dataclass(slots=True)
class GuardrailVerificationReport:
    """Verification report for a guardrail bundle."""

    current_verification: VerificationResult


@dataclass(slots=True)
class ProvenanceVerificationReport:
    """Verification report for a provenance bundle."""

    current_verification: VerificationResult


@dataclass(slots=True)
class InferenceBundle:
    """Proof-friendly representation of an inference transition."""

    provider: str
    before_json: str
    after_json: str
    recorded_proof: ZkProof

    def verify(self) -> InferenceVerificationReport:
        expected_hash = json_hash(
            {"before": self.before_json, "after": self.after_json, "provider": self.provider}
        )
        valid = expected_hash == self.recorded_proof.payload_hash
        return InferenceVerificationReport(
            current_verification=VerificationResult(valid=valid, details=expected_hash),
            verification_matches_recorded=valid,
        )

    def to_dict(self) -> JsonDict:
        return {
            "provider": self.provider,
            "before_json": self.before_json,
            "after_json": self.after_json,
            "recorded_proof": self.recorded_proof.to_dict(),
        }


@dataclass(slots=True)
class GuardrailBundle:
    """Proof-friendly representation of a plan guardrail claim."""

    provider: str
    actions: list[JsonDict]
    claim_type: str
    plan_digest: str

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def verify(self) -> GuardrailVerificationReport:
        valid = bool(self.actions) and bool(self.plan_digest)
        return GuardrailVerificationReport(
            current_verification=VerificationResult(
                valid=valid,
                details=self.plan_digest,
            )
        )

    def to_dict(self) -> JsonDict:
        return {
            "provider": self.provider,
            "actions": self.actions,
            "claim_type": self.claim_type,
            "plan_digest": self.plan_digest,
        }


@dataclass(slots=True)
class ProvenanceBundle:
    """Proof-friendly representation of provenance metadata for a world state."""

    provider: str
    source_label: str
    timestamp: int
    state_hash: str

    def verify(self) -> ProvenanceVerificationReport:
        valid = bool(self.source_label) and self.timestamp > 0 and bool(self.state_hash)
        return ProvenanceVerificationReport(
            current_verification=VerificationResult(
                valid=valid,
                details=self.state_hash,
            )
        )

    def to_dict(self) -> JsonDict:
        return {
            "provider": self.provider,
            "source_label": self.source_label,
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
        }


class MockVerifier:
    """Deterministic verifier used in tests and offline environments."""

    def verify_inference_bundle(self, bundle: InferenceBundle) -> InferenceVerificationReport:
        return bundle.verify()

    def verify_guardrail_bundle(self, bundle: GuardrailBundle) -> GuardrailVerificationReport:
        return bundle.verify()

    def verify_provenance_bundle(self, bundle: ProvenanceBundle) -> ProvenanceVerificationReport:
        return bundle.verify()


class ZkVerifier(MockVerifier):
    """Migration-safe verifier facade."""


def prove_inference_transition_bundle(
    before_json: str,
    after_json: str,
    *,
    provider: str = "mock",
) -> InferenceBundle:
    """Create an inference bundle from serialized world snapshots."""

    proof = ZkProof(
        backend="Mock",
        claim_type="InferenceTransition",
        payload_hash=json_hash({"before": before_json, "after": after_json, "provider": provider}),
    )
    return InferenceBundle(
        provider=provider,
        before_json=before_json,
        after_json=after_json,
        recorded_proof=proof,
    )


__all__ = [
    "GuardrailBundle",
    "GuardrailVerificationReport",
    "InferenceBundle",
    "InferenceVerificationReport",
    "MockVerifier",
    "ProvenanceBundle",
    "ProvenanceVerificationReport",
    "VerificationResult",
    "ZkProof",
    "ZkVerifier",
    "prove_inference_transition_bundle",
]
