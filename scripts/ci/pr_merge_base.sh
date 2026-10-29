#!/usr/bin/env bash
# Resolve the actual PR target, including maintained runtime branches.
set -euo pipefail
base_ref="${1:-main}"
git check-ref-format "refs/heads/$base_ref" >/dev/null
git merge-base "refs/remotes/origin/$base_ref" HEAD
