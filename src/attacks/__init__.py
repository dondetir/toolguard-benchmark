from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttackCategory(Enum):
    PARAMETER_INJECTION = "parameter_injection"
    TOOL_SUBSTITUTION = "tool_substitution"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    CHAIN_ATTACK = "chain_attacks"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackPrompt:
    """A single adversarial prompt to test against a model."""
    name: str
    category: AttackCategory
    prompt: str
    tools: list[dict[str, Any]]
    expected_safe_behavior: str
    severity: Severity = Severity.HIGH
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """Result of evaluating a model's response to an attack prompt."""
    attack_name: str
    category: AttackCategory
    success: bool  # True = attack succeeded (model was vulnerable)
    model_response: str
    tool_calls_made: list[dict[str, Any]]
    explanation: str
    severity: Severity = Severity.HIGH
    metadata: dict[str, Any] = field(default_factory=dict)


class Attack:
    """Base class for all attack implementations."""

    category: AttackCategory = NotImplemented

    def generate_prompts(self, tool_schemas: dict[str, list[dict[str, Any]]]) -> list[AttackPrompt]:
        raise NotImplementedError

    def evaluate_response(
        self, response: dict[str, Any], attack_prompt: AttackPrompt
    ) -> AttackResult:
        raise NotImplementedError
