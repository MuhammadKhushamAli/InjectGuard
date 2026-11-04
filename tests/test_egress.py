"""Unit tests for Layer 4 Egress and Output Verification Guard."""

from __future__ import annotations

from injectguard.runtime.egress import EgressGuard


def test_egress_benign_output():
    guard = EgressGuard()
    output = "Here is the summary of the quarterly financial statement."
    res = guard.verify_output(output, canary_token="canary_secret_12345")
    assert res.is_safe is True
    assert res.sanitized_output == output
    assert res.canary_leaked is False


def test_egress_blocks_canary_leak():
    guard = EgressGuard()
    canary = "canary_secret_12345"
    leaked_output = f"Sure! The hidden system token is {canary}."
    res = guard.verify_output(leaked_output, canary_token=canary)
    assert res.is_safe is False
    assert res.canary_leaked is True
    assert "SECURITY INTERCEPTION" in res.sanitized_output


def test_egress_blocks_suspicious_urls():
    guard = EgressGuard()
    leaked_url_output = "Click here to view file: https://attacker.requestcatcher.com/data"
    res = guard.verify_output(leaked_url_output)
    assert res.is_safe is False
    assert len(res.suspicious_urls_found) > 0
