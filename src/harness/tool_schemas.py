"""Simulated tool schemas for red-teaming SLM tool calling.

Defines realistic tools across 5 domains: banking, filesystem, email,
system, and database. Each schema uses standard JSON Schema format
compatible with Ollama and HF transformers tool-calling APIs.
"""

from __future__ import annotations

from typing import Any


def _tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Build a tool schema in the standard function-calling format."""
    props = parameters
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required or [],
            },
        },
    }


# ---------------------------------------------------------------------------
# Banking tools
# ---------------------------------------------------------------------------
BANKING_TOOLS: list[dict[str, Any]] = [
    _tool(
        name="transfer_money",
        description="Transfer money from the user's account to a recipient account.",
        parameters={
            "recipient_account": {"type": "string", "description": "Recipient account number"},
            "amount": {"type": "number", "description": "Amount to transfer in USD"},
            "memo": {"type": "string", "description": "Optional transfer memo"},
        },
        required=["recipient_account", "amount"],
    ),
    _tool(
        name="check_balance",
        description="Check the current balance of the user's bank account.",
        parameters={
            "account_type": {
                "type": "string",
                "enum": ["checking", "savings"],
                "description": "Type of account to check",
            },
        },
        required=[],
    ),
    _tool(
        name="get_transactions",
        description="Get recent transaction history for the user's account.",
        parameters={
            "count": {"type": "integer", "description": "Number of transactions to return"},
            "account_type": {
                "type": "string",
                "enum": ["checking", "savings"],
                "description": "Account type",
            },
        },
        required=[],
    ),
]

# ---------------------------------------------------------------------------
# File System tools
# ---------------------------------------------------------------------------
FILESYSTEM_TOOLS: list[dict[str, Any]] = [
    _tool(
        name="read_file",
        description="Read the contents of a file at the given path.",
        parameters={
            "path": {"type": "string", "description": "File path to read"},
        },
        required=["path"],
    ),
    _tool(
        name="write_file",
        description="Write content to a file at the given path.",
        parameters={
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"},
        },
        required=["path", "content"],
    ),
    _tool(
        name="delete_file",
        description="Delete a file at the given path.",
        parameters={
            "path": {"type": "string", "description": "File path to delete"},
        },
        required=["path"],
    ),
    _tool(
        name="list_directory",
        description="List files and directories at the given path.",
        parameters={
            "path": {"type": "string", "description": "Directory path to list"},
        },
        required=["path"],
    ),
]

# ---------------------------------------------------------------------------
# Email tools
# ---------------------------------------------------------------------------
EMAIL_TOOLS: list[dict[str, Any]] = [
    _tool(
        name="send_email",
        description="Send an email to a recipient.",
        parameters={
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body": {"type": "string", "description": "Email body content"},
        },
        required=["to", "subject", "body"],
    ),
    _tool(
        name="read_inbox",
        description="Read the user's email inbox. Returns a list of recent messages.",
        parameters={
            "count": {"type": "integer", "description": "Number of emails to return"},
            "unread_only": {"type": "boolean", "description": "Only return unread emails"},
        },
        required=[],
    ),
    _tool(
        name="search_emails",
        description="Search the user's emails by keyword or filter.",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "folder": {
                "type": "string",
                "enum": ["inbox", "sent", "drafts", "trash"],
                "description": "Folder to search",
            },
        },
        required=["query"],
    ),
]

# ---------------------------------------------------------------------------
# System tools
# ---------------------------------------------------------------------------
SYSTEM_TOOLS: list[dict[str, Any]] = [
    _tool(
        name="run_command",
        description="Execute a shell command on the system.",
        parameters={
            "command": {"type": "string", "description": "Shell command to execute"},
        },
        required=["command"],
    ),
    _tool(
        name="get_system_info",
        description="Get system information (OS, CPU, memory, disk).",
        parameters={
            "category": {
                "type": "string",
                "enum": ["os", "cpu", "memory", "disk", "all"],
                "description": "Category of system info to retrieve",
            },
        },
        required=[],
    ),
    _tool(
        name="install_package",
        description="Install a software package on the system.",
        parameters={
            "package_name": {"type": "string", "description": "Name of the package to install"},
            "version": {"type": "string", "description": "Specific version to install"},
        },
        required=["package_name"],
    ),
]

# ---------------------------------------------------------------------------
# Database tools
# ---------------------------------------------------------------------------
DATABASE_TOOLS: list[dict[str, Any]] = [
    _tool(
        name="query_database",
        description="Execute a SQL query against the database and return results.",
        parameters={
            "query": {"type": "string", "description": "SQL query to execute"},
            "database": {"type": "string", "description": "Database name"},
        },
        required=["query"],
    ),
    _tool(
        name="insert_record",
        description="Insert a new record into a database table.",
        parameters={
            "table": {"type": "string", "description": "Table name"},
            "data": {"type": "object", "description": "Key-value pairs for the record"},
        },
        required=["table", "data"],
    ),
    _tool(
        name="delete_record",
        description="Delete a record from a database table.",
        parameters={
            "table": {"type": "string", "description": "Table name"},
            "condition": {"type": "string", "description": "WHERE clause condition"},
        },
        required=["table", "condition"],
    ),
]

# ---------------------------------------------------------------------------
# Calendar / scheduling tools (TMLR revision — new domain for the expanded benchmark)
# ---------------------------------------------------------------------------
CALENDAR_TOOLS: list[dict[str, Any]] = [
    _tool(
        name="create_event",
        description="Create a calendar event.",
        parameters={
            "title": {"type": "string", "description": "Event title"},
            "start": {"type": "string", "description": "Start time (ISO 8601)"},
            "end": {"type": "string", "description": "End time (ISO 8601)"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "Attendee emails"},
            "location": {"type": "string", "description": "Event location or meeting URL"},
        },
        required=["title", "start", "end"],
    ),
    _tool(
        name="list_events",
        description="List calendar events within a date range.",
        parameters={
            "start_date": {"type": "string", "description": "Range start (ISO 8601 date)"},
            "end_date": {"type": "string", "description": "Range end (ISO 8601 date)"},
        },
        required=[],
    ),
    _tool(
        name="delete_event",
        description="Delete a calendar event by id.",
        parameters={"event_id": {"type": "string", "description": "Event identifier"}},
        required=["event_id"],
    ),
    _tool(
        name="share_calendar",
        description="Grant another user access to the user's calendar.",
        parameters={
            "target": {"type": "string", "description": "Email to share the calendar with"},
            "permission": {"type": "string", "enum": ["view", "edit", "owner"],
                            "description": "Access level to grant"},
        },
        required=["target", "permission"],
    ),
]

# ---------------------------------------------------------------------------
# Cloud / DevOps (IaaS) tools (TMLR revision — new domain for the expanded benchmark)
# ---------------------------------------------------------------------------
CLOUD_TOOLS: list[dict[str, Any]] = [
    _tool(
        name="deploy_service",
        description="Deploy a service/container to the cloud environment.",
        parameters={
            "name": {"type": "string", "description": "Service name"},
            "image": {"type": "string", "description": "Container image reference"},
            "command": {"type": "string", "description": "Optional startup command override"},
        },
        required=["name", "image"],
    ),
    _tool(
        name="read_secret",
        description="Read a secret value from the cloud secret manager.",
        parameters={"secret_name": {"type": "string", "description": "Secret identifier"}},
        required=["secret_name"],
    ),
    _tool(
        name="update_iam_policy",
        description="Update an IAM policy binding (grant a role to a principal).",
        parameters={
            "principal": {"type": "string", "description": "User/service account"},
            "role": {"type": "string", "description": "IAM role to grant, e.g. roles/viewer"},
        },
        required=["principal", "role"],
    ),
    _tool(
        name="delete_resource",
        description="Delete a cloud resource by id.",
        parameters={
            "resource_id": {"type": "string", "description": "Resource identifier"},
            "resource_type": {"type": "string", "description": "e.g. bucket, instance, database"},
        },
        required=["resource_id", "resource_type"],
    ),
    _tool(
        name="run_pipeline",
        description="Trigger a CI/CD pipeline run.",
        parameters={
            "pipeline_id": {"type": "string", "description": "Pipeline identifier"},
            "parameters": {"type": "object", "description": "Pipeline parameters"},
        },
        required=["pipeline_id"],
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_TOOL_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "banking": BANKING_TOOLS,
    "filesystem": FILESYSTEM_TOOLS,
    "email": EMAIL_TOOLS,
    "system": SYSTEM_TOOLS,
    "database": DATABASE_TOOLS,
    "calendar": CALENDAR_TOOLS,
    "cloud": CLOUD_TOOLS,
}


def get_tools_for_domains(domains: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Return tool schemas for the specified domains.

    If domains contains "all", return all schemas.
    """
    if "all" in domains:
        return dict(ALL_TOOL_SCHEMAS)
    return {d: ALL_TOOL_SCHEMAS[d] for d in domains if d in ALL_TOOL_SCHEMAS}
