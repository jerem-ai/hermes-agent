"""Operator-installed passive contracts; run with python -m hermes_cli.acceptance_contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from agent.acceptance_contract import bind_verifier, directory, load, register, report
from agent.acceptance_verification import verify


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
    agent = SimpleNamespace(session_id=args.session)
    if args.action == "attach":
        sha = register(args.session, json.loads(args.spec.read_bytes()))
        print(json.dumps({"contract": sha, "automatic_continuations": False}))
        return
    bound = load(agent)
    if bound is None:
        parser.error("session has no acceptance contract")
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


if __name__ == "__main__":
    main()
