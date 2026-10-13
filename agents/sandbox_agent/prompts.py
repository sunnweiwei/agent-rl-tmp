from agents.core.tool_calls import tools_prompt

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the task workspace (the current directory) and return its output. "
                "Each call runs in a fresh process: shell state such as `cd` or exported variables does not "
                "persist, but files written to the workspace do. Non-zero exit codes are reported as errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to run."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Run a Python script in the task workspace and return its stdout. Use print() to see values. "
                "Each call is a fresh interpreter; persist intermediate data in workspace files."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "The Python source code to run."}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the workspace."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file in the workspace. Parent directories are created.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace."},
                    "content": {"type": "string", "description": "The full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_file",
            "description": "Replace the first occurrence of `old_str` with `new_str` in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace."},
                    "old_str": {"type": "string", "description": "Exact text to replace; must occur in the file."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_answer",
            "description": (
                "Finish the task. Call this exactly once, when you are done. For questions, pass only the final "
                "answer value; for tasks that modify the workspace, pass a one-line summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer."}},
                "required": ["answer"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an autonomous agent working in a sandboxed Linux workspace. The harness parses your XML
function calls, executes them, returns the observations, and continues until you call return_answer.

Rules:
- Every turn must contain at least one function call until you call return_answer.
- Inspect before acting: look at files and intermediate results instead of guessing.
- Verify your result (run it, test it, re-check the numbers) before calling return_answer.
- Use at most {max_calls_per_turn} function calls per turn.

{tools}"""


def build_system_prompt(tools: list[dict], max_calls_per_turn: int) -> str:
    return SYSTEM_PROMPT.format(max_calls_per_turn=max_calls_per_turn, tools=tools_prompt(tools))
