# hydra-agent

A prototype: route each message to a topic-specific, isolated LLM
conversation ("shard") instead of growing one context and evicting from it.

Ask about the weather → a "weather" shard is created. Ask an unrelated
coding question → a new "code" shard is created; the weather shard is
untouched. Ask about the weather again → routed back to the *original*
weather shard, full history intact, no summarization or eviction ever
happened.

See [`docs/design/orchestration-idea.md`](docs/design/orchestration-idea.md)
for what this is, why it might help, its real tradeoffs, and the prior art
this is based on (AWS Agent Squad, ContextBranch, SWRouter, and more). See
[`docs/architecture.md`](docs/architecture.md) for how it's actually built,
including a real bug this project ran into (and how it was fixed) with the
`deepseek-flash` reasoning model.

This project has **no dependency on any other codebase** — it stands
entirely on its own.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

export DEEPSEEK_API_KEY=sk-...   # https://platform.deepseek.com

.venv/bin/pytest -q              # unit tests, no network calls

.venv/bin/hydra chat             # or: .venv/bin/python -m hydra chat
```

## CLI reference

```
hydra chat [--session NAME] [--router-model MODEL] [--shard-model MODEL] [--data-dir DIR] [--verbose]
```

- `--session` (default `default`) — a named session; each name gets its own
  persisted set of shards under `<data-dir>/sessions/<name>/`. Reuse a name
  to resume exactly where you left off, including across restarts.
- `--router-model` / `--shard-model` (default `deepseek-flash` for both) —
  which model handles routing decisions vs. actual shard conversations.
- `--data-dir` (default `.hydra`) — where session state is stored on disk.
- `--verbose` / `-v` — auto-print every thread's full stats table after
  every turn, not just when you ask for it with `:status`. Toggle it on or
  off mid-session with `:verbose` instead of restarting.

Inside the chat REPL:

```
:status            show every thread's id, topic, turns, and context size
:verbose           toggle auto-printing full thread stats after every turn
:switch <id>       force the NEXT message to a specific existing thread
:new <label>       force the NEXT message to start a brand-new thread
:help              show this help
:quit / :q         exit (Ctrl-D also exits)
```

`:status` (and any turn while `--verbose`/`:verbose` is on) shows an
*extended* table beyond the compact per-turn line: every thread's turn
count, last real input- and output-token counts, the global turn it was
created on, and how many turns it's been since that thread was last used —
the same "since active" figure the router itself sees when deciding where
to route the next message, made visible instead of only ever going into a
model prompt.

Every turn prints, in color: whether the router routed to an existing
thread or spawned a new one (and what that decision's own call cost in
tokens), the reply from that thread's own conversation, and a compact
`ctx~N tok (real)` line — the *actual* token count DeepSeek reported for
that thread's last call, not an estimate.

## Project layout

```
hydra/            the package: router, shards, persistence, CLI (see
                   docs/architecture.md for what each module does)
tests/            unit tests against a scripted fake LLM client -- zero
                   network calls, zero API key required
docs/design/      the idea: what/why/advantages/disadvantages/research
docs/             architecture.md: how it's actually built
```

## Known limitations (see `docs/design/orchestration-idea.md` for more)

- Each shard is **fully isolated** — there's no shared/global memory across
  shards in this version. A fact learned in one thread is invisible to
  every other thread.
- Misrouting is possible: the router can plausibly (not just by outright
  hallucination) pick the wrong existing thread for an ambiguous message.
  It never *trusts* an invalid answer (see `docs/architecture.md`'s
  "never guess" discipline), but a plausible wrong guess isn't detectable
  from inside the system.
- A single thread that itself runs very long will still eventually need its
  own context management — routing sits on top of that problem, it doesn't
  solve it.
- Session state is plain JSON files with last-writer-wins persistence; no
  protection against two processes writing the same session concurrently.
