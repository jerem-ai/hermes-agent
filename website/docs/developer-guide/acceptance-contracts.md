# Passive acceptance contracts

An acceptance contract keeps the original objective and its checks outside the conversation summary. It runs no model calls, background controller or automatic continuation. The operator installs reviewed criteria for a specific session. Descendant sessions resolve the same contract through session lineage.

Each task has an ID, objective, type, optional parent and explicit checks. Implementation tasks require behavior acceptance. Delivery tasks also require runtime acceptance. Completing a documentation child leaves its unfinished parent open. An undefined verifier remains pending.

Use the profile's existing Python runtime and home:

~~~sh
python -m hermes_cli.acceptance_contract --session SESSION attach --spec contract.json
python -m hermes_cli.acceptance_contract --session SESSION status
python -m hermes_cli.acceptance_contract --session SESSION verify --task TASK --check CHECK
~~~

The contract schema is hermes.acceptance-contract.v1, with root and a tasks array. Check kinds are documentation, tests, behavior and runtime. Each check contains id, description, kind and an optional verifier.

A verifier defines argv, files (absolute paths to SHA-256 values), optional cwd, timeout_seconds and maximum_age_seconds. Pin its program and the source/artifacts whose behavior it verifies. Choose a bounded, read-only acceptance program. It must return JSON containing "passed": true and exit zero. Commands run directly without a shell. Before delivering a completion claim, the harness refreshes the registered verification programs. Installing a verifier authorizes that bounded read-only check at this boundary; no inference or model-control program belongs here. At most eight checks and 30 seconds of total verifier time are allowed.

Receipts bind the contract, task, check, evidence kind and verifier definition. Missing, failed, expired or source-mismatched receipts leave the check open. A failed verification attempt invalidates its previous receipt before execution. Raw command output stays out of the receipt.

To install a reviewed verifier later, the operator can amend that check while preserving every objective and requirement:

~~~sh
python -m hermes_cli.acceptance_contract --session SESSION bind-verifier \
  --task TASK --check CHECK --expected-contract CURRENT_SHA --verifier verifier.json
~~~

The previous immutable contract is retained. Existing receipts must be verified again against the new contract. Attaching another objective to an already bound session is refused; use a new conversation for a new parent objective.

The parent and pending checks are added through the current user context and restored after compression. The cached system prompt is unchanged. Contracted sessions buffer text delivery until the final response has passed the guard. Broad completion claims with outstanding checks become an explicit open-parent report. A verified child may report completion while explicitly stating that its parent remains open.

The prose detector covers a finite set of completion claims. The durable task status always derives from the checks. This is a completion discipline feature within the existing user account; it does not create a security boundary against code that can rewrite the account's own state.
