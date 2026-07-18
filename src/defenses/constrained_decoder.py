"""Constrained decoding safety layer for tool-calling.

Provides post-hoc validation of tool calls against a security policy.
The policy is loaded from a YAML file and enforces parameter constraints
and intent-based tool permissions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SafetyConstraint:
    """A single constraint to enforce during tool call generation."""
    name: str
    description: str
    check: str  # "blocklist", "allowlist", "range", "pattern"
    # Tools this constraint applies to (empty = all tools)
    tools: list[str] = field(default_factory=list)
    # Parameter names this constraint checks (empty = all params)
    parameters: list[str] = field(default_factory=list)
    # Check-specific parameters (pattern, values, min, max, etc.)
    params: dict[str, Any] = field(default_factory=dict)
    # "block" = block if matched; "require_match" = block if NOT matched
    action: str = "block"


@dataclass
class IntentPattern:
    """Maps a user-intent regex to allowed/blocked tool sets."""
    pattern: str
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)


@dataclass
class ConstraintViolation:
    """Record of a constraint being violated."""
    constraint_name: str
    tool_name: str
    parameter_name: str
    value: Any
    explanation: str


class ConstrainedDecoder:
    """Safety layer that validates tool calls against a security policy.

    Supports four check types:
      - blocklist:     block if the value contains any listed string
      - allowlist:     block if the value is not in the allowed set
      - range:         block if the numeric value is outside [min, max]
      - pattern:       action="block" blocks if matched;
                       action="require_match" blocks if NOT matched

    Additionally supports intent-based tool permission checks: given the
    original user prompt, ensure the tool called is appropriate for what
    the user asked.
    """

    def __init__(self) -> None:
        self.constraints: list[SafetyConstraint] = []
        self.intent_patterns: list[IntentPattern] = []

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    @classmethod
    def load_policy(cls, yaml_path: str | Path) -> "ConstrainedDecoder":
        """Load a security policy from a YAML file and return a configured decoder."""
        decoder = cls()
        with open(yaml_path) as f:
            policy = yaml.safe_load(f)

        # Load parameter constraints
        for entry in policy.get("constraints", []):
            constraint = SafetyConstraint(
                name=entry["name"],
                description=entry.get("description", entry["name"]),
                check=entry["check"],
                tools=entry.get("tools", []),
                parameters=entry.get("parameters", []),
                params=entry.get("params", {}),
                action=entry.get("action", "block"),
            )
            decoder.constraints.append(constraint)

        # Load tool-permission intent patterns (top-level key)
        tool_perms = policy.get("tool_permissions", {})
        for ip_entry in tool_perms.get("intent_patterns", []):
            ip = IntentPattern(
                pattern=ip_entry["pattern"],
                allowed_tools=ip_entry.get("allowed_tools", []),
                blocked_tools=ip_entry.get("blocked_tools", []),
            )
            decoder.intent_patterns.append(ip)

        return decoder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: SafetyConstraint) -> None:
        self.constraints.append(constraint)

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_prompt: str | None = None,
    ) -> list[ConstraintViolation]:
        """Validate a tool call against all registered constraints.

        Args:
            tool_name:   The name of the tool being called.
            arguments:   The arguments dict passed to the tool.
            user_prompt: The original user message (optional). When provided,
                         intent_patterns are checked to verify the tool is
                         consistent with the user's stated intent.

        Returns:
            A list of ConstraintViolation objects. An empty list means the
            call is considered safe.
        """
        violations: list[ConstraintViolation] = []

        # 1. Check parameter constraints
        for constraint in self.constraints:
            # Skip if this constraint targets specific tools and this tool isn't one of them
            if constraint.tools and tool_name not in constraint.tools:
                continue

            for param_name, value in arguments.items():
                # Skip if this constraint targets specific parameters and this param isn't one
                if constraint.parameters and param_name not in constraint.parameters:
                    continue

                violation = self._check_constraint(constraint, tool_name, param_name, value)
                if violation:
                    violations.append(violation)

        # 2. Check intent-based tool permissions
        if user_prompt is not None:
            intent_violations = self._check_intent(tool_name, user_prompt)
            violations.extend(intent_violations)

        return violations

    def is_blocked(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_prompt: str | None = None,
    ) -> bool:
        """Return True if the tool call should be blocked."""
        return len(self.validate_tool_call(tool_name, arguments, user_prompt)) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_constraint(
        self,
        constraint: SafetyConstraint,
        tool_name: str,
        param_name: str,
        value: Any,
    ) -> ConstraintViolation | None:
        """Check a single constraint against a parameter value."""
        if constraint.check == "blocklist":
            blocked = constraint.params.get("values", [])
            value_str = str(value).lower()
            for b in blocked:
                if b.lower() in value_str:
                    return ConstraintViolation(
                        constraint_name=constraint.name,
                        tool_name=tool_name,
                        parameter_name=param_name,
                        value=value,
                        explanation=f"Blocked value '{b}' found in parameter '{param_name}'",
                    )

        elif constraint.check == "allowlist":
            allowed = constraint.params.get("values", [])
            value_str = str(value)
            if value_str not in [str(a) for a in allowed]:
                return ConstraintViolation(
                    constraint_name=constraint.name,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    value=value,
                    explanation=f"Value '{value}' is not in the allowed set",
                )

        elif constraint.check == "pattern":
            pattern = constraint.params.get("pattern", "")
            value_str = str(value)
            action = constraint.action

            if action == "require_match":
                # Block if the value does NOT match the pattern
                if not re.search(pattern, value_str):
                    return ConstraintViolation(
                        constraint_name=constraint.name,
                        tool_name=tool_name,
                        parameter_name=param_name,
                        value=value,
                        explanation=(
                            f"Value '{value}' does not match required pattern "
                            f"(allow_safe_paths: only /home/user/, /tmp/, or relative paths allowed)"
                        ),
                    )
            else:
                # action == "block": block if the value DOES match the pattern
                if re.search(pattern, value_str):
                    return ConstraintViolation(
                        constraint_name=constraint.name,
                        tool_name=tool_name,
                        parameter_name=param_name,
                        value=value,
                        explanation=(
                            f"Value '{value}' matches blocked pattern "
                            f"for constraint '{constraint.name}'"
                        ),
                    )

        elif constraint.check == "range":
            min_val = constraint.params.get("min")
            max_val = constraint.params.get("max")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if min_val is not None and numeric < min_val:
                return ConstraintViolation(
                    constraint_name=constraint.name,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    value=value,
                    explanation=f"Value {value} below minimum {min_val}",
                )
            if max_val is not None and numeric > max_val:
                return ConstraintViolation(
                    constraint_name=constraint.name,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    value=value,
                    explanation=f"Value {value} above maximum {max_val}",
                )

        return None

    def _check_intent(
        self, tool_name: str, user_prompt: str
    ) -> list[ConstraintViolation]:
        """Check that the tool called is consistent with the user's intent."""
        violations: list[ConstraintViolation] = []
        for ip in self.intent_patterns:
            if re.search(ip.pattern, user_prompt):
                if tool_name in ip.blocked_tools:
                    violations.append(
                        ConstraintViolation(
                            constraint_name="tool_intent_mismatch",
                            tool_name=tool_name,
                            parameter_name="(tool selection)",
                            value=tool_name,
                            explanation=(
                                f"Tool '{tool_name}' is not permitted for the detected intent. "
                                f"Intent pattern matched: '{ip.pattern}'. "
                                f"Allowed tools: {ip.allowed_tools}"
                            ),
                        )
                    )
        return violations
