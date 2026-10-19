---
title: Execution targets
---

`agent.execution_targets` lets an operator name known machines and project workspaces in a profile.
An explicit, single-host operational request such as “Check the widget repository on my laptop”
then constrains the built-in terminal and filesystem calls for that task. It does not open an SSH
connection or grant access. The normal tool approval and command checks still apply.

```yaml
agent:
  execution_targets:
    - id: laptop
      aliases: [laptop, dev-laptop]
      destination: dev-laptop
      projects:
        widget: /home/operator/workspaces/widget
    - id: server
      aliases: [server]
      local: true
```

The destination names an existing SSH configuration alias. The remote project path belongs to that
machine. For this example, a supported terminal call is:

```bash
ssh -o BatchMode=yes dev-laptop 'git -C /home/operator/workspaces/widget log -1'
```

The guard accepts a single SSH invocation and a quoted remote shell string. The remote command must
start with `git -C <workspace>` or `cd <workspace> &&` when a project workspace is bound. Transport
options such as `BatchMode` and `ConnectTimeout` are supported; destination-overriding options are
refused. A matching native SSH terminal backend can also use the normal file tools within that
workspace. Local `hostname`, `pwd` and `whoami` probes remain available but do not verify the remote
repository. Code execution and delegation cannot substitute a local filesystem for the selected host.

Rejected calls return a correction before execution. Three target refusals halt the turn through the
existing tool guardrail. A text answer with no successful target command is replaced with an explicit
unverified result and a failed turn status. Streaming and output transformations use that same check.
Existing provider failures and cancellation retain their own status. A successful command verifies
where the tool ran; it does not establish that the model interpreted its output correctly.

The target survives follow-ups, compression and native session lineage through a profile-owned
`execution-targets` directory. These JSON records contain configuration hashes and target/project IDs;
they contain no prompt or generated answer. Verification success is reset on each turn. A late worker
cannot verify a newer turn. Changes to the target declarations invalidate saved bindings until the
operator checks the change and the user restates the host. An explicit request naming a configured
local target clears the remote constraint. Multi-host requests use ordinary tool behavior rather
than inferring a single destination. Explicit topic resets such as “change gears” also clear it.
This is a task-quality guard, not an account security boundary.

To inspect the records with the profile selected in the managed runtime:

```python
import json
from hermes_constants import get_hermes_home

for path in (get_hermes_home() / "execution-targets").glob("*.json"):
    print(path.name, json.loads(path.read_bytes()))
```

A healthy active binding has the current declaration hash, a configured target ID and, when selected,
a configured project ID. A local or multi-host selection can intentionally leave no remote target.
The feature is disabled when the profile declares no targets. Alias detection covers configured names
in operational requests; quoted reply text and fenced examples do not establish a new target.
