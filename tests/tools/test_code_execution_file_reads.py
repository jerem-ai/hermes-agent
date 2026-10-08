"""Python RPC reads return current bytes without losing chat read deduplication."""
import json

import pytest


@pytest.mark.parametrize("kernel_mode", ["per_call", "session"])
def test_python_reads_after_chat_read_return_content(tmp_path, monkeypatch, kernel_mode):
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from model_tools import handle_function_call
    from tools.code_kernel import shutdown_all_kernels

    monkeypatch.setenv("TERMINAL_ENV", "local")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"code_execution:\n  mode: strict\n  kernel_mode: {kernel_mode}\n"
        "  timeout: 30\n  max_tool_calls: 12\nterminal:\n  env_type: local\n"
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"label": "Crème 雪", "items": ["-0.5", "2.125", "0.375"]}))
    original = inventory.read_bytes()
    output = tmp_path / "summary.json"
    protected = home / "auth.json"
    protected.write_text('{"test": "protected"}')
    home_token = set_hermes_home_override(home)
    cwd_token = set_session_cwd(str(tmp_path))
    task_id = "python-read-" + kernel_mode
    try:
        first = json.loads(handle_function_call("read_file", {"path": str(inventory)}, task_id=task_id))
        assert "content" in first
        code = f'''
import json, re
from decimal import Decimal
from hermes_tools import read_file, write_file
for _ in range(5):
    raw = read_file({str(inventory)!r})["content"]
data = json.loads(re.sub(r"(?m)^\\s*\\d+\\|", "", raw))
summary = {{"count": len(data["items"]), "total": str(sum(map(Decimal, data["items"]), Decimal("0"))), "label": data["label"]}}
write_file({str(output)!r}, json.dumps(summary, ensure_ascii=False))
saved = read_file({str(output)!r})["content"]
print(saved)
blocked = read_file({str(protected)!r})
assert "content" not in blocked and "error" in blocked
'''
        result = json.loads(handle_function_call(
            "execute_code", {"code": code}, task_id=task_id,
            enabled_tools=["read_file", "write_file"],
        ))
        assert result["status"] == "success", result
        assert json.loads(output.read_text()) == {"count": 3, "total": "2.000", "label": "Crème 雪"}
        assert inventory.read_bytes() == original
        # The host chat still gets an unchanged stub after the Python cell.
        chat = json.loads(handle_function_call("read_file", {"path": str(inventory)}, task_id=task_id))
        assert chat.get("content_returned") is False and "content" not in chat
        # A real file change must be visible to a later cell, including a reused kernel.
        inventory.write_text('{"label":"fresh","items":["7"]}')
        again = json.loads(handle_function_call(
            "execute_code", {"code": f"from hermes_tools import read_file\nprint(read_file({str(inventory)!r})['content'])"},
            task_id=task_id, enabled_tools=["read_file", "write_file"],
        ))
        assert again["status"] == "success" and '"fresh"' in again["output"], again
    finally:
        shutdown_all_kernels()
        reset_session_cwd(cwd_token)
        reset_hermes_home_override(home_token)


def test_python_read_scope_clears_after_dispatch_failure(tmp_path, monkeypatch):
    import model_tools
    from tools.code_execution_rpc import _default_dispatch

    monkeypatch.setenv("TERMINAL_ENV", "local")
    path = tmp_path / "read.txt"
    path.write_text("unchanged\n")
    task_id = "rpc-read-error"
    real = model_tools.handle_function_call
    assert "content" in json.loads(real("read_file", {"path": str(path)}, task_id=task_id))

    def fail(*args, **kwargs):
        raise RuntimeError("dispatch failed")

    with monkeypatch.context() as context:
        context.setattr(model_tools, "handle_function_call", fail)
        with pytest.raises(RuntimeError, match="dispatch failed"):
            _default_dispatch(task_id)("read_file", {"path": str(path)})
    chat = json.loads(real("read_file", {"path": str(path)}, task_id=task_id))
    assert chat.get("content_returned") is False and "content" not in chat
