"""Top-level orchestration: register Headroom MCP across detected agents."""

from __future__ import annotations

import sys
from collections.abc import Iterable

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec
from .claude import ClaudeRegistrar
from .codex import CodexRegistrar
from .opencode import OpencodeRegistrar

#: Default proxy URL used when none is given.
DEFAULT_PROXY_URL = "http://127.0.0.1:8787"
HEADROOM_SERVER_NAME = "headroom"
HEADROOM_MODULE = "headroom.cli"
HEADROOM_MCP_ARGS = ("-m", HEADROOM_MODULE, "mcp", "serve")
HEADROOM_PROXY_URL_ENV = "HEADROOM_PROXY_URL"


def get_all_registrars() -> list[MCPRegistrar]:
    """Return one instance of every registrar implemented today.

    The list grows as we add adapters for Cursor, Continue, Cline, etc.
    """
    return [ClaudeRegistrar(), CodexRegistrar(), OpencodeRegistrar()]


def build_headroom_spec(proxy_url: str = DEFAULT_PROXY_URL) -> ServerSpec:
    """Construct the canonical :class:`ServerSpec` for the headroom server.

    The spec is identical across agents — every JSON/TOML registrar
    serializes the same shape into its own format.
    """
    env: dict[str, str] = {}
    if proxy_url and proxy_url != DEFAULT_PROXY_URL:
        env[HEADROOM_PROXY_URL_ENV] = proxy_url
    return ServerSpec(
        name=HEADROOM_SERVER_NAME,
        command=sys.executable,
        args=HEADROOM_MCP_ARGS,
        env=env,
    )


def build_serena_spec(context: str) -> ServerSpec:
    """Construct the canonical Serena MCP server spec for an agent context.

    ``--open-web-dashboard False`` suppresses Serena's browser popup on
    startup. Headroom installs Serena by default, so without this flag every
    wrapped session opens the Serena dashboard tab even for users who never
    opted into Serena or created a ``~/.serena/serena_config.yml``. The flag
    overrides Serena's own config at startup (it sets
    ``web_dashboard_open_on_launch=False``), so it works regardless of the
    user's local config. The dashboard backend still runs and remains
    reachable at http://localhost:24282/dashboard/ for anyone who wants it —
    only the automatic browser-open is disabled.
    """
    return ServerSpec(
        name="serena",
        command="uvx",
        args=(
            "--from",
            "git+https://github.com/oraios/serena",
            "serena",
            "start-mcp-server",
            "--project-from-cwd",
            "--context",
            context,
            "--open-web-dashboard",
            "False",
        ),
    )


def install_everywhere(
    proxy_url: str = DEFAULT_PROXY_URL,
    *,
    agents: Iterable[str] | None = None,
    force: bool = False,
    registrars: Iterable[MCPRegistrar] | None = None,
) -> dict[str, RegisterResult]:
    """Install the headroom MCP server into every detected agent.

    Args:
        proxy_url: URL the MCP server should contact for retrieval.
        agents: If given, only install into agents whose ``name`` matches.
        force: Pass through to each registrar — overwrites mismatched config.
        registrars: Inject a custom registrar list (test seam).

    Returns:
        Dict keyed by registrar name. Includes :attr:`RegisterStatus.NOT_DETECTED`
        entries for agents we know about that aren't installed locally.
    """
    spec = build_headroom_spec(proxy_url)
    selected = list(registrars) if registrars is not None else get_all_registrars()

    if agents is not None:
        agent_set = set(agents)
        selected = [r for r in selected if r.name in agent_set]

    results: dict[str, RegisterResult] = {}
    for registrar in selected:
        if not registrar.detect():
            results[registrar.name] = RegisterResult(
                RegisterStatus.NOT_DETECTED,
                f"{registrar.display_name} not found on this system",
            )
            continue
        results[registrar.name] = registrar.register_server(spec, force=force)

    return results
