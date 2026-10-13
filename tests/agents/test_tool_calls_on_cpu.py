import pytest

from agents.core.tool_calls import parse_tool_calls, tools_prompt
from agents.sandbox_agent.agent import DEFAULTS, _truncate, match_score, resolve_config
from agents.sandbox_agent.prompts import TOOLS


def test_parse_single_call():
    text = "Look first.\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>"
    assert parse_tool_calls(text) == [{"function": "bash", "arguments": {"command": "ls -la"}}]


def test_parse_multiple_adjacent_calls():
    text = (
        "<function=read_file>\n<parameter=path>a.txt</parameter>\n</function>\n"
        "<function=read_file>\n<parameter=path>b.txt</parameter>\n</function>"
    )
    assert [c["arguments"]["path"] for c in parse_tool_calls(text)] == ["a.txt", "b.txt"]


def test_parse_keeps_only_last_block():
    text = (
        "Example: <function=bash>\n<parameter=command>rm -rf /</parameter>\n</function>\n\n\n\n\n"
        "Now for real:\n<function=bash>\n<parameter=command>pwd</parameter>\n</function>"
    )
    assert parse_tool_calls(text) == [{"function": "bash", "arguments": {"command": "pwd"}}]


def test_parse_ignores_reasoning():
    text = "<think><function=bash><parameter=command>x</parameter></function></think>no calls"
    assert parse_tool_calls(text) == []


def test_parse_preserves_code_indentation():
    code = "for i in range(3):\n    print(i)"
    text = f"<function=execute_python>\n<parameter=code>\n{code}\n</parameter>\n</function>"
    assert parse_tool_calls(text)[0]["arguments"]["code"] == code


def test_parse_recovers_inline_parameter():
    text = "<function=write_file>\n<parameter=path=out.txt</parameter>\n<parameter=content>hi</parameter>\n</function>"
    assert parse_tool_calls(text)[0]["arguments"] == {"path": "out.txt", "content": "hi"}


@pytest.mark.parametrize("text", [None, "", "plain answer", "<function=bash>unterminated"])
def test_parse_no_calls(text):
    assert parse_tool_calls(text) == []


def test_tools_prompt_lists_every_tool():
    prompt = tools_prompt(TOOLS)
    for tool in TOOLS:
        assert f": {tool['function']['name']} ----" in prompt
    assert "(1) command (string, required)" in prompt


def test_resolve_config_precedence_and_types():
    config = resolve_config(
        {"max_turns": "50", "process_rewards_enabled": "true", "trajectory_trace_path": ""},
        {"max_turns": 7, "tool_timeout": "30"},
    )
    assert config["max_turns"] == 7
    assert config["tool_timeout"] == 30.0
    assert config["process_rewards_enabled"] is True
    assert config["trajectory_trace_path"] == DEFAULTS["trajectory_trace_path"]


def test_truncate_keeps_head_and_tail():
    text = "a" * 100 + "b" * 100
    out = _truncate(text, 30)
    assert out.startswith("a" * 20) and out.endswith("b" * 10)
    assert "170 characters truncated" in out
    assert _truncate("short", 30) == "short"


@pytest.mark.parametrize(
    ("answer", "truth", "score"),
    [("18", "18", 1.0), (" 1,000. ", "1000", 1.0), ("Paris", "paris", 1.0), ("", "1", 0.0)],
)
def test_match_score_exact(answer, truth, score):
    assert match_score(answer, truth) == score
