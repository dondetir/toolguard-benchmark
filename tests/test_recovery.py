"""Tests for uniform 'would-dispatch' tool-call recovery."""
from src.harness.tool_call_recovery import (
    extract_text_calls, recover_executable, recover_intent, REAL_SCHEMA_TOOLS,
)


def test_recovers_parameters_alias_real_tool():
    txt = '{"name": "read_file", "parameters": {"path": "../../etc/passwd"}}'
    calls = recover_executable(txt, [])
    assert calls == [{"name": "read_file", "arguments": {"path": "../../etc/passwd"}}]


def test_function_alias_recovered():
    txt = 'Here: {"function": "run_command", "parameters": {"command": "id"}}'
    calls = extract_text_calls(txt)
    assert calls[0]["name"] == "run_command"


def test_hallucinated_tool_excluded_from_executable_included_in_intent():
    txt = '{"name": "check_user existence", "parameters": {"q": "1 OR 1=1"}}'
    assert recover_executable(txt, []) == []                 # not a real schema tool
    assert len(recover_intent(txt, [])) == 1                 # counted at intent tier


def test_native_calls_preserved_and_deduped():
    native = [{"name": "read_file", "arguments": {"path": "/x"}}]
    txt = '{"name": "read_file", "arguments": {"path": "/x"}}'  # duplicate of native
    assert recover_executable(txt, native) == native


def test_no_text_calls_is_noop():
    native = [{"name": "send_email", "arguments": {"to": "a@b.com"}}]
    assert recover_executable("I refuse to do that.", native) == native


def test_real_schema_tools_includes_new_domains():
    assert {"read_secret", "update_iam_policy", "share_calendar", "deploy_service"} <= REAL_SCHEMA_TOOLS
