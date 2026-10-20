"""Classic CLI memory commands; learning status never loads a mutable store."""

def handle_memory_command(cli, cmd: str):
    """Handle /memory slash command — pending review + approval-gate toggle."""
    from hermes_cli.cli_commands_mixin import _t
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    args = cmd.strip().split()[1:]
    from hermes_cli.learning_status import handle_slash
    status = handle_slash(args)
    if status is not None:
        print(status)
        return
    store = getattr(cli.agent, "_memory_store", None) if getattr(cli, "agent", None) else None
    if store is None:
        # No live agent store (e.g. Desktop GUI): use a fresh on-disk store, as the gateway
        # does — same MEMORY/USER.md, same configured char limits.
        # Apply against a freshly loaded on-disk store, mirroring the gateway path
        # (gateway/slash_commands.py): it persists to the same MEMORY/USER.md and creates MEMORY.md on
        # the first approved write. Without this the shared handler returns "memory store unavailable".
        # See #46783.
        from tools.memory_tool import load_on_disk_store
        store = load_on_disk_store()
    out = handle_pending_subcommand(
        wa.MEMORY, args, memory_store=store,
        set_mode_fn=lambda enabled: cli._save_write_approval("memory", enabled))
    print(out if out is not None else _t("memory.unknown_subcommand"))

