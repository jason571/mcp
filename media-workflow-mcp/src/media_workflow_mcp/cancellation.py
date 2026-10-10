"""SDK cancellation identity, kept separate from media workflow policy."""
from collections.abc import Callable
from mcp.server.mcpserver import Context


def peer_cancellation_check(ctx: Context) -> Callable[[], bool]:
    # MCP Python 2.x's high-level Context omits BaseContext.cancel_requested.
    # Its request-scoped ServerSession retains the same DispatchContext. Read
    # that SDK event, not exception text, HTTP disconnects or a global flag.
    # Keep this compatibility boundary isolated and fail before submitting any
    # work if the SDK no longer provides the per-request cancellation signal.
    channel = getattr(ctx.session, "_request_outbound", None)
    event = getattr(channel, "cancel_requested", None)
    if event is None or not callable(getattr(event, "is_set", None)):
        raise RuntimeError("MCP SDK does not expose request-scoped cancellation")
    return event.is_set
