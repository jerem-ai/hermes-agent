"""Operator-installed passive contracts; run with python -m hermes_cli.acceptance_contract."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

from agent.acceptance_contract import bind_verifier, directory, load, register, report
from agent.acceptance_verification import verify



@contextmanager
def session_agent(session_id):
    """Lineage lookup is read-only and stays inside the owning profile."""
    from hermes_state import SessionDB
    path = directory().parent / "state.db"
    db = SessionDB(db_path=path, read_only=True) if path.exists() else None
    try:
        yield SimpleNamespace(session_id=session_id, _session_db=db)
    finally:
        if db is not None:
            db.close()


def operate(args, agent):
    bound = load(agent)
    if bound is None:
        raise ValueError("session has no acceptance contract")
    if args.action == "status":
        print(json.dumps(report(agent), ensure_ascii=False, indent=2))
        return
    if args.action == "bind-verifier":
        sha = bind_verifier(agent, args.expected_contract, args.task, args.check,
                            json.loads(args.verifier.read_bytes()))
        print(json.dumps({"contract": sha, "previous_contract": args.expected_contract}))
        return
    spec, sha = bound
    receipt = verify(directory(agent), spec, sha, args.task, args.check)
    print(json.dumps(receipt))
    raise SystemExit(0 if receipt["passed"] else 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    attach = sub.add_parser("attach")
    attach.add_argument("--spec", type=Path, required=True)
    sub.add_parser("status")
    check = sub.add_parser("verify")
    check.add_argument("--task", required=True)
    check.add_argument("--check", required=True)
    amend = sub.add_parser("bind-verifier", help="Operator action: bind reviewed verification code")
    amend.add_argument("--task", required=True)
    amend.add_argument("--check", required=True)
    amend.add_argument("--expected-contract", required=True)
    amend.add_argument("--verifier", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "attach":
        sha = register(args.session, json.loads(args.spec.read_bytes()))
        print(json.dumps({"contract": sha, "automatic_continuations": False}))
        return
    with session_agent(args.session) as agent:
        operate(args, agent)


if __name__ == "__main__":
    main()
