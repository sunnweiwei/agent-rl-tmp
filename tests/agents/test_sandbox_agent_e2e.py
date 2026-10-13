"""End-to-end: sandbox server (unsafe subprocess backend) + client + sandbox agent loop.

The agent-loop test drives a scripted LLM through a real chat template; set
AGENT_TEST_TOKENIZER to a local Qwen tokenizer directory to run it.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from agents.core.agent import Completion, TaskContext
from agents.sandbox_agent.agent import agent_loop
from agents.sandbox_agent.client import SandboxClient, SandboxError

REPO = Path(__file__).resolve().parents[2]
TOKEN = "test-token"


def _make_tasks(root: Path) -> None:
    task = root / "demo" / "sum"
    (task / "workspace").mkdir(parents=True)
    (task / "workspace" / "numbers.txt").write_text("3 4 5\n10\n")
    (task / "verify.py").write_text("def verify(answer):\n    return (answer == '22', 'expected 22')\n")
    fix = root / "demo" / "fix"
    (fix / "workspace").mkdir(parents=True)
    (fix / "tests").mkdir()
    (fix / "workspace" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (fix / "tests" / "check.py").write_text(
        "import sys\nsys.path.insert(0, '.')\nfrom calc import add\nassert add(2, 3) == 5\n"
    )
    (fix / "task.yaml").write_text(
        "setup_commands:\n  - echo ready > marker.txt\n"
        "reward:\n  type: command\n  command: python3 .eval/tests/check.py\n"
    )


@pytest.fixture(scope="module")
def sandbox_url(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sandbox")
    _make_tasks(tmp / "tasks")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "SANDBOX_ALLOW_UNSAFE_SUBPROCESS": "1",
        "SANDBOX_ENABLED_PROFILES": "nsjail",
        "SANDBOX_TASK_ROOT": str(tmp / "tasks"),
        "SANDBOX_WORKSPACE_ROOT": str(tmp / "ws"),
        "SANDBOX_API_TOKEN": TOKEN,
        "SANDBOX_CONFIG": str(tmp / "missing.yaml"),
        "PYTHONPATH": str(REPO),
    }
    log = open(tmp / "server.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "sandbox_server.server:app", "--port", str(port)],
        env=env,
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{url}/v1/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        server.kill()
        pytest.fail((tmp / "server.log").read_text())
    yield url
    server.terminate()
    server.wait()


def _live_trials(url):
    capacity = requests.get(f"{url}/v1/capacity", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    return sum(c["live_trials"] for c in capacity)


def test_requires_token(sandbox_url):
    assert requests.post(f"{sandbox_url}/v1/sessions", json={"profile": "nsjail"}).status_code == 401


@pytest.mark.asyncio
async def test_client_session_trial_actions_reward(sandbox_url):
    client = SandboxClient(sandbox_url, api_token=TOKEN)
    session = await client.open_session("nsjail")
    trial = await session.create_trial("demo/sum")
    assert await trial.execute("bash", {"command": "cat numbers.txt"}) == "3 4 5\n10\n"
    assert await trial.execute("execute_python", {"code": "print(3 + 4 + 5 + 10)"}) == "22\n"
    with pytest.raises(SandboxError, match="command exited 1"):
        await trial.execute("bash", {"command": "cat missing.txt"})
    with pytest.raises(SandboxError, match="escapes"):
        await trial.execute("read_file", {"path": "../../etc/passwd"})
    assert (await trial.reward(answer="22"))["passed"] is True
    assert (await trial.reward(answer="21"))["passed"] is False
    await trial.destroy()

    trial = await session.create_trial("demo/fix")
    assert await trial.execute("read_file", {"path": "marker.txt"}) == "ready\n"
    assert (await trial.reward())["passed"] is False
    await trial.execute("str_replace_file", {"path": "calc.py", "old_str": "a - b", "new_str": "b + a  # fixed"})
    assert (await trial.reward())["passed"] is True
    await trial.destroy()
    await session.close()
    assert _live_trials(sandbox_url) == 0


def _call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"Next step.\n<function={name}>\n{body}</function>"


class ScriptedLLM:
    def __init__(self, tokenizer, responses):
        self.tokenizer = tokenizer
        self.responses = list(responses)
        self.prompts = []

    async def create_completion(self, prompt_ids, **_):
        self.prompts.append(list(prompt_ids))
        if not self.responses:
            return None
        text = self.responses.pop(0)
        ids = self.tokenizer.encode(text, add_special_tokens=False) + [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        return Completion(content=text, token_ids=ids, log_probs=[-0.1] * len(ids))


def _context(tokenizer, llm, options, is_train=True):
    from omegaconf import OmegaConf

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"prompt_length": 4096, "response_length": 8192}}})
    agent_loop_stub = SimpleNamespace(
        loop=asyncio.get_running_loop(), apply_chat_template_kwargs={"enable_thinking": False}
    )
    return TaskContext(
        config=config,
        global_step=0,
        is_train=is_train,
        tokenizer=tokenizer,
        agent_loop=agent_loop_stub,
        llm_client=llm,
        sampling_params={},
        agent_options=options,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AGENT_TEST_TOKENIZER"), reason="set AGENT_TEST_TOKENIZER to a Qwen tokenizer dir")
async def test_agent_loop_end_to_end(sandbox_url, monkeypatch, tmp_path):
    from transformers import AutoTokenizer

    monkeypatch.setenv("SANDBOX_API_TOKEN", TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(os.environ["AGENT_TEST_TOKENIZER"])
    options = {
        "sandbox_url": sandbox_url,
        "process_rewards_enabled": "true",
        "context_penalty_max": 0.5,
        "context_penalty_buffer_tokens": 1024,
        "trajectory_trace_path": str(tmp_path / "trace.jsonl"),
    }
    llm = ScriptedLLM(
        tokenizer,
        [
            _call("bash", command="cat numbers.txt"),
            "The answer is 22.",  # no function call -> parse error
            _call("execute_python", code="print(sum(int(x) for x in open('numbers.txt').read().split()))"),
            _call("bash", command="cat missing.txt") + "\n" + _call("read_file", path="numbers.txt"),
            _call("return_answer", answer="22"),
        ],
    )
    data = {
        "raw_prompt": [{"role": "user", "content": "What is the sum of the numbers in numbers.txt?"}],
        "data_source": "sandbox_agent",
        "extra_info": {"task_id": "demo/sum"},
    }
    out = await agent_loop(data, _context(tokenizer, llm, options))
    metrics = out.extra_fields["reward_extra_info"]

    assert out.reward_score == 1.0
    assert metrics["agent/terminated_return_answer"] == 1
    assert (metrics["agent/tool_calls"], metrics["agent/parse_errors"], metrics["agent/tool_errors"]) == (4, 1, 1)
    # One parse-error turn (-0.25) and one tool-error turn (-0.05).
    assert metrics["agent/process_reward_actions"] == 2
    assert metrics["agent/process_reward_sum"] == pytest.approx(-0.30)
    assert any(out.process_advantages)
    assert len(out.response_ids) == len(out.response_mask) == len(out.response_logprobs) == len(out.process_advantages)
    # Each generation request extends the previous context exactly (prefix-cache friendly).
    for before, after in zip(llm.prompts, llm.prompts[1:], strict=False):
        assert after[: len(before)] == before
    assert (tmp_path / "trace.jsonl").read_text().count("\n") == 1

    # Ground-truth reward on a scratch trial; validation applies no shaping.
    llm = ScriptedLLM(tokenizer, ["no call", _call("return_answer", answer="18")])
    data = {
        "raw_prompt": [{"role": "user", "content": "Compute 3*6."}],
        "data_source": "sandbox_agent",
        "reward_model": {"ground_truth": "18"},
        "extra_info": {},
    }
    out = await agent_loop(data, _context(tokenizer, llm, options, is_train=False))
    assert out.reward_score == 1.0
    assert out.extra_fields["reward_extra_info"]["agent/process_reward_actions"] == 0
    assert _live_trials(sandbox_url) == 0
