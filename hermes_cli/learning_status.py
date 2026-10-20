"""Read-only learning configuration and recorded evidence. No provider initialization."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

from agent.learning_review_events import EVENTS, TERMINAL_EVENTS, event_path
from agent.review_idle_queue import defer_mode
from hermes_constants import get_hermes_home

_TAIL_BYTES = 1024 * 1024


def _managed_netloc() -> str:
    from agent.auxiliary_client import _managed_local_netloc
    return _managed_local_netloc()


def _section(config: dict, name: str) -> dict:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _configuration(config: dict) -> dict:
    from utils import is_truthy_value

    memory, skills = _section(config, "memory"), _section(config, "skills")
    task = _section(_section(config, "auxiliary"), "background_review")
    model = _section(config, "model")
    mode = defer_mode(task)
    # This is a CONFIGURED endpoint, not the runtime of an already-running conversation.
    # An explicit different provider/model without an endpoint needs runtime resolution:
    # status must remain read-only, so leave that case unknown.
    base = task.get("base_url")
    inherits = (task.get("provider", "auto") in (None, "", "auto", model.get("provider"))
                and task.get("model", "") in (None, "", model.get("default")))
    if not base and inherits:
        base = model.get("base_url")
    dispatch = "unknown"
    if mode == "never":
        dispatch = "after_turn"
    elif base:
        try:
            netloc = urlsplit(str(base)).netloc.lower()
            dispatch = "idle_queue" if netloc and netloc == _managed_netloc() else "after_turn"
        except ValueError:
            pass
    enabled = is_truthy_value(task.get("enabled"), default=True)
    return {"background_review_enabled": enabled,
            "memory_enabled": is_truthy_value(memory.get("memory_enabled"), default=True),
            "user_profile_enabled": is_truthy_value(memory.get("user_profile_enabled"), default=True),
            "memory_review_turn_interval": memory.get("nudge_interval", 10),
            "skill_review_tool_iteration_interval": skills.get("creation_nudge_interval", 10),
            "defer_mode": mode, "defer_scope": "managed_local_runtime_only",
            "configured_dispatch": dispatch if enabled else "disabled",
            "live_session_counters": "not_observed"}


def _read_events(home: Path, session_id: str | None) -> tuple[list[dict], int, bool]:
    rows, invalid, bounded = [], 0, False
    path = event_path(home)
    for source in (path.with_suffix(".jsonl.1"), path):
        if not source.exists():
            continue
        try:
            with source.open("rb") as stream:
                size = source.stat().st_size
                if size > _TAIL_BYTES:
                    stream.seek(-_TAIL_BYTES, 2)
                    stream.readline()
                    bounded = True
                lines = stream.read().splitlines()
            for line in lines:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get("event") not in EVENTS:
                        raise ValueError("invalid event")
                    if not session_id or row.get("session_id") == session_id:
                        rows.append(row)
                except (ValueError, TypeError):
                    invalid += 1
        except OSError:
            invalid += 1
    return rows, invalid, bounded


def _usage_receipts(home: Path, session_id: str | None) -> tuple[int | None, str]:
    path = home / "state.db"
    if not path.exists():
        return 0, "no_database"
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
            query = "SELECT count(*) FROM session_model_usage WHERE task = ?"
            params = ["background_review"]
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            return conn.execute(query, params).fetchone()[0], "available"
    except sqlite3.Error:
        return None, "unavailable"


def collect_status(home: Path, config: dict, *, session_id: str | None = None, limit: int = 10) -> dict:
    home = Path(home)
    rows, invalid, bounded = _read_events(home, session_id)
    outcomes = Counter(r["event"] for r in rows if r["event"] in TERMINAL_EVENTS)
    writes = {key: 0 for key in ("memory", "skill", "staged")}
    for row in rows:
        counts = row.get("writes")
        if isinstance(counts, dict):
            for key in writes:
                value = counts.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    writes[key] += value
    receipts, usage_state = _usage_receipts(home, session_id)
    # Only export the body-free event schema, even if a file contains extra fields.
    recent = [{k: r[k] for k in ("time", "session_id", "review_id", "event", "reason", "error_type") if k in r}
              for r in rows[-max(1, min(limit, 100)):]]
    return {"version": 1, "profile_home": str(home), "session_id": session_id,
            "configuration": _configuration(config),
            "observed": {"outcome_evidence": "recorded" if rows else "not_recorded",
                         "outcomes": dict(outcomes), "successful_write_results": writes,
                         "usage_receipts": receipts, "usage_evidence": usage_state,
                         "unreadable_records": invalid, "bounded_history": bounded, "recent": recent},
            "limits": ["Configuration does not prove a review ran.",
                       "Usage receipts do not prove completion or a saved lesson.",
                       "Events do not prove live liveness, lesson quality, or reuse.",
                       "Existing sessions keep their frozen memory snapshot."]}


def cmd_status(args) -> None:
    from hermes_cli.config import load_config_readonly

    result = collect_status(get_hermes_home(), load_config_readonly(),
                            session_id=getattr(args, "session_id", None), limit=getattr(args, "limit", 10))
    print(render_status(result, as_json=getattr(args, "json", False)))


def render_status(result: dict, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(result, ensure_ascii=False, indent=2)
    cfg, observed = result["configuration"], result["observed"]
    lines = [f"Learning status for {result['profile_home']}",
             f"  Automatic review configured: {'enabled' if cfg['background_review_enabled'] else 'disabled'}",
             f"  Memory trigger: {cfg['memory_review_turn_interval']} user turns",
             f"  Skill trigger: {cfg['skill_review_tool_iteration_interval']} tool iterations",
             f"  Configured dispatch: {cfg['configured_dispatch']} (defer={cfg['defer_mode']})",
             f"  Recorded outcomes: {observed['outcomes'] or 'none recorded'}",
             f"  Successful write results: {observed['successful_write_results']}",
             f"  Review usage receipts: {observed['usage_receipts']} ({observed['usage_evidence']})"]
    for row in observed["recent"]:
        lines.append(f"  {row.get('time', '?')} {row['event']} session={row.get('session_id', '?')}")
    if observed["unreadable_records"] or observed["bounded_history"]:
        lines.append("  Evidence is partial; inspect --json for coverage details.")
    lines.append("  Recorded events do not establish current liveness, lesson quality, or reuse.")
    return "\n".join(lines)


def handle_slash(args: list[str]) -> str | None:
    """Shared, read-only /memory status before callers load any mutable memory store."""
    if not args or args[0].lower() != "status":
        return None
    from hermes_cli.config import load_config_readonly

    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--session", dest="session_id")
    parser.add_argument("--limit", type=int, default=10)
    try:
        options, unknown = parser.parse_known_args(args[1:])
        if unknown:
            raise ValueError("unknown option")
    except (argparse.ArgumentError, ValueError):
        return "Usage: /memory status [--json] [--session ID] [--limit 1..100]"
    result = collect_status(get_hermes_home(), load_config_readonly(), session_id=options.session_id, limit=options.limit)
    return render_status(result, as_json=options.json)
