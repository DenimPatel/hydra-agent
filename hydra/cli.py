"""The interactive REPL: `hydra chat` (or `python -m hydra chat`).

Every turn prints, in color: the router's decision (routed to an existing
shard, or spawned a new one), the reply, and a compact context-size line for
the shard that just handled it. `/status` prints a full table of every
shard's state. Nothing here decides routing itself -- that's router.py; this
module only renders decisions and drives the read-persist-print loop.
"""

import argparse
import sys

from rich.console import Console
from rich.table import Table

from . import store
from .config import Config
from .llm import DeepSeekClient
from .router import Router

_PALETTE = ["cyan", "green", "yellow", "magenta", "blue",
            "bright_cyan", "bright_green", "bright_yellow", "bright_magenta"]

HELP_TEXT = """\
/status            show every thread's id, topic, turns, and context size
/verbose           toggle auto-printing full thread stats after every turn
/switch <id>       force the NEXT message to a specific existing thread
/new <label>       force the NEXT message to start a brand-new thread
/help              show this help
/quit / /q         exit (Ctrl-D also exits)"""


def _color_for(shard_id, known_ids):
    """Deterministic: the same shard_id always gets the same color across a
    session, based on its position among all known ids sorted -- not on
    call order, so it doesn't shift as new shards are added.
    """
    ordered = sorted(known_ids)
    index = ordered.index(shard_id) if shard_id in ordered else 0
    return _PALETTE[index % len(_PALETTE)]


def status_rows(shards, turn, extended=False):
    """The plain data behind `/status`, separate from its rendering so the
    values can be asserted directly in tests instead of substring-matched
    against a whole printed table. Each row is
    [id, topic, turns, ctx tok, created turn] plus, when extended,
    [out tok (last), turns since active].
    """
    rows = []
    for shard in sorted(shards.values(), key=lambda s: s.shard_id):
        since_active = max(0, turn - (shard.created_turn + shard.turns))
        row = [shard.shard_id, shard.topic_label, str(shard.turns),
               str(shard.context_tokens()), str(shard.created_turn)]
        if extended:
            row.append(str(shard.last_output_tokens))
            row.append(str(since_active))
        rows.append(row)
    return rows


class Session(object):
    """Owns the Router plus persistence -- the CLI's only stateful object."""

    def __init__(self, client, config, verbose=False):
        self.client = client
        self.config = config
        self.session_dir = config.session_dir()
        shards, turn = store.load_all_shards(self.session_dir)
        self.router = Router(client, config, shards=shards, turn=turn)
        self.forced_shard_id = None
        self.forced_new_label = None
        self.verbose = verbose

    def persist(self):
        store.save_registry(
            self.session_dir, self.router.turn,
            {sid: shard.topic_label for sid, shard in self.router.shards.items()},
        )
        for shard in self.router.shards.values():
            store.save_shard(self.session_dir, shard)

    def handle_message(self, console, user_text):
        if self.forced_shard_id is not None:
            shard = self.router.shards[self.forced_shard_id]
            self.router.turn += 1
            self._print_routed(console, shard, forced=True)
            self.forced_shard_id = None
        elif self.forced_new_label is not None:
            shard = self.router.spawn(self.forced_new_label)
            self._print_new(console, shard, forced=True)
            self.forced_new_label = None
        else:
            shard, decision = self.router.route(user_text)
            if decision.is_new:
                self._print_new(console, shard, cost_tokens=decision.cost_tokens)
            else:
                self._print_routed(console, shard, cost_tokens=decision.cost_tokens)

        reply = shard.send(self.client, self.config.shard_model, user_text,
                            max_tokens=self.config.shard_max_tokens)
        color = _color_for(shard.shard_id, self.router.shards.keys())
        console.print("[bold %s]%s (%s)[/] %s" % (
            color, shard.topic_label, shard.shard_id, reply))
        console.print("[dim]  ctx~%d tok (real) | turn %d for this thread[/]" % (
            shard.context_tokens(), shard.turns))
        if self.verbose:
            self.print_status(console, extended=True)
        self.persist()

    def _print_new(self, console, shard, cost_tokens=0, forced=False):
        color = _color_for(shard.shard_id, self.router.shards.keys())
        tag = "FORCED NEW" if forced else "ROUTER -> NEW"
        cost = "" if forced else " (router call: %d tok)" % cost_tokens
        console.print("[bold white on magenta] %s [/] spawned [bold %s]%s (%s)[/]%s" % (
            tag, color, shard.topic_label, shard.shard_id, cost))

    def _print_routed(self, console, shard, cost_tokens=0, forced=False):
        color = _color_for(shard.shard_id, self.router.shards.keys())
        tag = "FORCED" if forced else "ROUTER"
        cost = "" if forced else " (router call: %d tok)" % cost_tokens
        console.print("[bold white on blue] %s [/] -> existing [bold %s]%s (%s)[/]%s" % (
            tag, color, shard.topic_label, shard.shard_id, cost))

    def print_status(self, console, extended=False):
        """`extended=True` (used automatically when `self.verbose` is on, or
        always via `/status` -- see below) adds the columns that don't fit
        in the compact table: each shard's last OUTPUT token count (input is
        already in the base table as "ctx tok"), and how many global turns
        it's been since that shard was last used -- the same "since active"
        figure the router itself sees on its own menu (`router._render_menu`),
        made visible here instead of only ever going into a model prompt.
        """
        title = "hydra session: %s" % self.config.session_name
        table = Table(title=title + (" (verbose)" if extended else ""))
        table.add_column("id")
        table.add_column("topic")
        table.add_column("turns", justify="right")
        table.add_column("ctx tok (last call)", justify="right")
        table.add_column("created turn", justify="right")
        if extended:
            table.add_column("out tok (last)", justify="right")
            table.add_column("turns since active", justify="right")
        if not self.router.shards:
            console.print("[dim]no threads yet -- send a message to create the first one[/]")
            return
        for row in status_rows(self.router.shards, self.router.turn, extended):
            color = _color_for(row[0], self.router.shards.keys())
            table.add_row("[%s]%s[/]" % (color, row[0]),
                          "[%s]%s[/]" % (color, row[1]),
                          *row[2:])
        console.print(table)


def _build_parser():
    parser = argparse.ArgumentParser(prog="hydra")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="start (or resume) an interactive session")
    chat.add_argument("--session", default="default", help="session name (default: 'default')")
    chat.add_argument("--router-model", default="deepseek-flash")
    chat.add_argument("--shard-model", default="deepseek-flash")
    chat.add_argument("--data-dir", default=".hydra")
    chat.add_argument("--verbose", "-v", action="store_true",
                       help="auto-print every thread's full stats table after each turn "
                            "(toggle mid-session with /verbose)")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()

    config = Config(router_model=args.router_model, shard_model=args.shard_model,
                     data_dir=args.data_dir, session_name=args.session)
    try:
        client = DeepSeekClient()
    except RuntimeError as exc:
        console.print("[bold red]error:[/] %s" % exc)
        return 1

    try:
        session = Session(client, config, verbose=args.verbose)
    except store.RegistryError as exc:
        console.print("[bold red]error:[/] could not load session: %s" % exc)
        return 1

    console.print("[bold]hydra-agent[/] -- session=%r router=%s shard=%s%s" % (
        args.session, config.router_model, config.shard_model,
        " [bold yellow](verbose)[/]" if session.verbose else ""))
    console.print("[dim]%d thread(s) loaded from %s. /help for commands, /quit to exit.[/]" % (
        len(session.router.shards), session.session_dir))

    while True:
        try:
            line = console.input("[bold white]you>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("")
            return 0
        text = (line or "").strip()
        if not text:
            continue
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()
        if cmd in ("/quit", "/q", "/exit"):
            return 0
        if cmd == "/help":
            console.print(HELP_TEXT)
            continue
        if cmd == "/status":
            session.print_status(console, extended=True)
            continue
        if cmd == "/verbose":
            session.verbose = not session.verbose
            console.print("[dim]verbose auto-status is now %s[/]" % (
                "ON" if session.verbose else "OFF"))
            continue
        if cmd == "/switch":
            if not arg:
                console.print("[red]usage: /switch <id> -- see /status[/]")
            elif arg not in session.router.shards:
                console.print("[red]no such thread %r -- see /status[/]" % arg)
            else:
                session.forced_shard_id = arg
                console.print("[dim]next message forced to %s[/]" % arg)
            continue
        if cmd == "/new":
            if not arg:
                console.print("[red]usage: /new <label>[/]")
            else:
                session.forced_new_label = arg
                console.print("[dim]next message will start a new thread %r[/]" % arg)
            continue
        if text.startswith("/"):
            console.print("[red]unknown command %r -- /help for the list[/]" % text)
            continue
        try:
            session.handle_message(console, text)
        except Exception as exc:  # a bad turn should not kill the whole session
            console.print("[bold red]error:[/] %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    sys.exit(main())
