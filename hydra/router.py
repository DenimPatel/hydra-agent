"""The Router: decides, for each incoming message, whether it continues an
existing shard's topic or starts a new one.

One tool-free LLM call per message: a short menu of the currently known
shards (id, topic label, turn count, turns since last active) plus the
newest user message, asking for exactly `EXISTING: <id>` or
`NEW: <topic label>`.

THE ONE DELIBERATE ASYMMETRY, worth reading before touching this file:
every failure mode -- the call raising, an unparseable reply, or a returned
id that isn't a real shard -- falls back to NEW. Spawning an unneeded shard
is a cheap, visible, always-recoverable mistake (worst case: two shards for
one topic, easily noticed in `:status`). Trusting a wrong EXISTING id would
silently splice this message into a different topic's history, corrupting
that shard in a way nothing surfaces until much later. When in doubt, split,
never merge.
"""

import re

from .shard import Shard

_VERDICT = re.compile(r"^\s*\**\s*(EXISTING|NEW)\s*:\s*(.+?)\s*\**\s*$",
                       re.IGNORECASE | re.MULTILINE)

DEFAULT_ROUTER_SYSTEM = (
    "You route messages in a multi-topic assistant to the right conversation "
    "thread. You are shown the threads that already exist (id, topic, how "
    "many turns, how long since last used) and the user's newest message. "
    "Decide: does this message continue one of the existing threads, or "
    "start a genuinely new topic?\n\n"
    "Reply with EXACTLY ONE line, one of:\n"
    "  EXISTING: <id>\n"
    "  NEW: <a short 2-4 word topic label>\n\n"
    "Prefer EXISTING whenever the message plausibly continues that thread's "
    "topic, even if other topics happened in between. Only say NEW when the "
    "message is not a good fit for any existing thread. If there are no "
    "existing threads yet, you must say NEW."
)


class RouteDecision(object):
    def __init__(self, is_new, shard_id=None, topic_label=None,
                 cost_tokens=0, reason=""):
        self.is_new = is_new
        self.shard_id = shard_id
        self.topic_label = topic_label
        self.cost_tokens = cost_tokens
        self.reason = reason


class Router(object):
    def __init__(self, client, config, shards=None, turn=0):
        self.client = client
        self.config = config
        self.shards = dict(shards or {})   # shard_id -> Shard
        self.turn = turn
        self._next_id = _next_shard_id(self.shards)

    def classify(self, user_text):
        """Read-only: decides where `user_text` should go, without mutating
        any state. Separated from `route()` so tests/CLI can inspect a
        decision before committing to it (e.g. for a `:status`-style dry run).
        """
        if not self.shards:
            return RouteDecision(is_new=True, topic_label=_fallback_label(user_text),
                                  reason="no existing threads yet")

        system = self.config.router_system_prompt or DEFAULT_ROUTER_SYSTEM
        menu = _render_menu(self.shards, self.turn, user_text)
        try:
            result = self.client.complete(
                model=self.config.router_model, system=system,
                messages=[{"role": "user", "content": menu}],
                max_tokens=self.config.router_max_tokens,
            )
        except Exception as exc:  # a broken router is a NEW thread, not a crash
            return RouteDecision(
                is_new=True, topic_label=_fallback_label(user_text),
                reason="router call failed: %s: %s" % (type(exc).__name__, exc))

        cost = (result.input_tokens or 0) + (result.output_tokens or 0)
        match = _VERDICT.search(result.text or "")
        if match is None:
            if not result.text and result.finish_reason == "length":
                reason = (
                    "router reply truncated to empty by max_tokens (a reasoning "
                    "model spent its whole budget on hidden reasoning) -- "
                    "raise router_max_tokens" )
            else:
                reason = "unparseable router reply: %r" % (result.text,)
            return RouteDecision(is_new=True, topic_label=_fallback_label(user_text),
                                  cost_tokens=cost, reason=reason)

        verdict, payload = match.group(1).upper(), match.group(2).strip()
        if verdict == "EXISTING":
            if payload in self.shards:
                return RouteDecision(is_new=False, shard_id=payload, cost_tokens=cost,
                                      reason="router: continues thread %s" % payload)
            return RouteDecision(
                is_new=True, topic_label=_fallback_label(user_text), cost_tokens=cost,
                reason="router named unknown thread %r -- never guessed, treated as NEW" % payload)
        return RouteDecision(is_new=True, topic_label=payload or _fallback_label(user_text),
                              cost_tokens=cost, reason="router: new topic")

    def route(self, user_text):
        """Decide AND commit: returns the Shard this message should go to,
        creating one if the decision was NEW.
        """
        decision = self.classify(user_text)
        if decision.is_new:
            shard = self.spawn(decision.topic_label)
        else:
            shard = self.shards[decision.shard_id]
            self.turn += 1
        return shard, decision

    def spawn(self, topic_label):
        """Public version of the NEW branch: create a fresh shard and advance
        the global turn counter. The CLI's `:new` path uses this instead of
        reaching into `_spawn_shard` and hand-rolling the turn bookkeeping.
        """
        shard = self._spawn_shard(topic_label)
        self.turn += 1
        return shard

    def _spawn_shard(self, topic_label):
        shard_id = "t%d" % self._next_id
        self._next_id += 1
        shard = Shard(shard_id=shard_id, topic_label=topic_label,
                       system_prompt=self.config.shard_system_prompt,
                       created_turn=self.turn)
        self.shards[shard_id] = shard
        return shard


def _next_shard_id(shards):
    """The next free `tN` id, derived from the HIGHEST existing id rather than
    the shard count. A missing or unreadable shard file makes a session's ids
    non-contiguous, and `len(shards) + 1` would then hand out an id that
    already exists -- silently overwriting (destroying) that shard.
    """
    highest = 0
    for shard_id in shards:
        if isinstance(shard_id, str) and shard_id.startswith("t") and shard_id[1:].isdigit():
            highest = max(highest, int(shard_id[1:]))
    return highest + 1


def _fallback_label(user_text):
    words = (user_text or "").split()[:4]
    return " ".join(words) or "untitled"


def _render_menu(shards, current_turn, user_text):
    lines = ["Existing threads (id | topic | turns | turns since active):"]
    for shard in sorted(shards.values(), key=lambda s: s.shard_id):
        since = max(0, current_turn - _last_turn(shard))
        lines.append("  %s | %s | %d | %d" % (
            shard.shard_id, shard.topic_label, shard.turns, since))
    lines.append("")
    lines.append("Newest user message: %s" % user_text)
    return "\n".join(lines)


def _last_turn(shard):
    # created_turn plus turns handled so far approximates "last active turn"
    # without needing a separate counter synced across saves.
    return shard.created_turn + shard.turns
