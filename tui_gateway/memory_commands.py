"""Profile-bound memory command helpers for Desktop and TUI."""
import contextlib

def pending_subcommand_store(session):
    """Memory store for the session's own profile: the live agent's store when there is one, else a
    freshly loaded on-disk store — mirroring cli.py's /memory handler and the gateway's (Desktop/TUI
    sessions have no long-lived agent store). ``None`` when even that fails, which the shared handler
    reports as "memory store unavailable" instead of claiming the write was applied."""
    from tui_gateway import server
    if session:
        with contextlib.suppress(Exception):
            store = getattr(session.get("agent"), "_memory_store", None)
            if store is not None:
                return store
    with contextlib.suppress(Exception):
        return server._tools_mod("tools.memory_tool").load_on_disk_store()
    return None

