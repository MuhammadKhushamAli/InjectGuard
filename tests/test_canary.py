"""Unit tests for Layer 4 Canary Token Manager."""

from __future__ import annotations

from injectguard.runtime.canary import CanaryManager


def test_canary_generation():
    manager = CanaryManager(token_prefix="canary_")
    token = manager.generate_token()
    assert token.startswith("canary_")
    assert len(token) > 16

    # Test uniqueness
    token2 = manager.generate_token()
    assert token != token2


def test_canary_prompt_injection():
    manager = CanaryManager()
    base_prompt = "You are a customer support agent."
    result = manager.inject_canary(base_prompt)

    assert result.canary_token in result.protected_system_prompt
    assert "STRICT CONFIDENTIALITY" in result.protected_system_prompt
    assert base_prompt in result.protected_system_prompt
