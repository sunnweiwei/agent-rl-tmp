"""XML function-call prompt and parser (the ``qwen3_coder`` call format).

Models emit calls as::

    <function=bash>
    <parameter=command>ls -la</parameter>
    </function>
"""

from __future__ import annotations

import re
from typing import Any

TOOL_CALL_FORMAT = """When calling tools, reply with one or more complete XML function blocks in this format,
with NO suffix after the last function block:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>.
- Parameters must be wrapped with <parameter=key>value</parameter>.
- Required parameters MUST be specified.
- You may include multiple function calls in one turn only when they are independent.
  They are executed from top to bottom.
- If a later action depends on an earlier tool result, wait for the next observation instead of guessing.
- Use only the functions listed above.
- You may provide optional reasoning before the first function call, but not after the final function call.
</IMPORTANT>"""

_FUNCTION_RE = re.compile(r"<function=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter(?:\s+name)?\s*=\s*[\"']?([A-Za-z_][\w-]*)[\"']?\s*>(.*?)</parameter>", re.DOTALL)
# Recovers the common malformed form <parameter=key=value</parameter>.
_INLINE_PARAM_RE = re.compile(r"<parameter=([A-Za-z_][\w-]*)=([^<>]*)</parameter>")


def describe_tools(tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, tool in enumerate(tools, start=1):
        fn = tool["function"]
        lines.append(f"---- BEGIN FUNCTION #{i}: {fn['name']} ----")
        lines.append(f"Description: {fn['description']}")
        properties = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        if properties:
            lines.append("Parameters:")
            for j, (name, info) in enumerate(properties.items(), start=1):
                status = "required" if name in required else "optional"
                desc = info.get("description", "No description provided")
                if "enum" in info:
                    desc += "\nAllowed values: [" + ", ".join(f"`{v}`" for v in info["enum"]) + "]"
                lines.append(f"  ({j}) {name} ({info.get('type', 'string')}, {status}): {desc}")
        else:
            lines.append("No parameters are required for this function.")
        lines.append(f"---- END FUNCTION #{i} ----")
    return "\n".join(lines)


def tools_prompt(tools: list[dict[str, Any]]) -> str:
    return f"You have access to the following functions:\n\n{describe_tools(tools)}\n\n{TOOL_CALL_FORMAT}"


def parse_tool_calls(text: str | None) -> list[dict[str, Any]]:
    """Return ``[{"function": name, "arguments": {...}}]`` for the last block of calls.

    Reasoning before ``</think>`` is ignored. Calls separated by more than a few
    lines start a new block, and only the final block is executed, so example
    calls quoted earlier in the response are not run.
    """
    if not text:
        return []
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    matches = list(_FUNCTION_RE.finditer(text))
    if not matches:
        return []
    groups = [[matches[0]]]
    for match in matches[1:]:
        if text.count("\n", groups[-1][-1].end(), match.start()) < 4:
            groups[-1].append(match)
        else:
            groups.append([match])
    return [{"function": m.group(1), "arguments": _parse_parameters(m.group(2))} for m in groups[-1]]


def _parse_parameters(body: str) -> dict[str, str]:
    arguments = {key: value.strip("\n") for key, value in _PARAM_RE.findall(body)}
    for key, value in _INLINE_PARAM_RE.findall(body):
        arguments.setdefault(key, value.strip())
    return arguments
