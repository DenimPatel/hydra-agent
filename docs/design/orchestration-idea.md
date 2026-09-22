# The idea: route by topic instead of evicting from one context

## What we're building

Most LLM agents keep one growing conversation: as it gets long, something
has to summarize or evict old material to stay within budget, and every
eviction strategy trades detail for space. `hydra-agent` tries a different
starting point: **unrelated topics never share a context in the first
place.**

A **Router** sits in front of every message. For each new message it decides:
does this continue a topic we already have an open thread for, or is it
something new? If it's a continuation, the message goes to that thread's
own, isolated conversation — picking up exactly where it left off, even if
several other topics happened in between. If it's new, a fresh thread
("shard") is spun up for it, with its own history, starting from nothing.

Concretely: ask about the weather → a "weather" shard is created. Ask an
unrelated coding question → a new "code" shard is created; the weather
shard is untouched, not summarized, not evicted, just not currently in use.
Ask about the weather again → the router recognizes this continues the
weather topic and routes back to the *original* weather shard, full history
intact, as if no interruption happened.

Each shard is a completely separate conversation as far as the model is
concerned — it never sees another shard's messages. The router's own job is
narrow and cheap: one short classification call per message, choosing
between the known shards (by id) or declaring a new topic.

## Advantages

- **No cross-topic context pollution.** A long tangent into an unrelated
  topic can't degrade the quality of a different, ongoing thread, because
  they're never in the same context window together.
- **No eviction overhead for switched-away topics.** Nothing needs to be
  summarized or tombstoned when a topic goes idle — its shard is simply not
  sent to the model again until it's needed. The full detail is always
  there when the topic recurs, not a lossy compressed version of it.
- **Real specialization becomes possible.** Each shard could eventually
  carry its own system prompt, tools, or even model choice suited to its
  topic, rather than one system prompt/toolset serving every kind of
  request.
- **This is a production-precedented pattern, not a novel guess.** AWS's
  [Agent Squad](https://awslabs.github.io/agent-squad/general/how-it-works/)
  framework runs almost exactly this shape at production scale: a
  classifier with a global view routes each turn to a specialized agent,
  and each agent only ever sees its own conversation history.

## Disadvantages

- **Misrouting is the central risk.** A wrong classification either
  needlessly fragments one topic into two threads, or — worse — could
  splice a message into the wrong thread's history if the router ever
  trusted a bad guess. `hydra-agent`'s router deliberately never trusts an
  unvalidated answer (see `docs/architecture.md`), but a *plausible*
  misclassification (the router picks a real, wrong existing thread) is
  still possible and not detectable from inside the system.
- **Cross-topic reasoning gets *harder*, not easier.** If someone asks
  "does that weather thing relate to the bug we found," no single shard
  has both contexts to draw the connection from. Isolating topics to avoid
  pollution also means giving up legitimate cross-references.
- **Shared facts need a separate mechanism — and this prototype doesn't
  have one.** A user's name, an ongoing preference, a project-wide
  constraint: none of these belong to one topic, but every shard would need
  them. In this version, **each shard is fully isolated with no shared
  memory at all** — a known, stated v1 limitation, not something silently
  glossed over.
- **This doesn't remove the need for eviction — it relocates it.** A single
  topic that itself runs very long (a multi-day project) will eventually
  need its own context management inside its shard. Routing is a sharding
  layer *on top of* that problem, not a replacement for it.
- **The router itself costs something on every turn.** One extra
  classification call per message, with its own latency, dollar cost, and
  chance of a wrong or malformed answer.

## Research findings

**Production frameworks:**

- [AWS Agent Squad](https://awslabs.github.io/agent-squad/general/how-it-works/)
  (formerly "multi-agent-orchestrator") is the closest production match: an
  LLM classifier routes each turn to one of several specialized agents. The
  classifier has a **global** view across every agent's history; each
  **individual agent only sees its own** conversation history, retrieved
  per user+session id. This validates the exact shape used here.
- [OpenAI Swarm / Agents SDK "handoffs"](https://openai.github.io/openai-agents-python/multi_agent/)
  solve a *different* problem: all agents in a "swarm" **share one
  conversation history** and transfer control to each other. That gives no
  context-isolation benefit — it's about who owns the next turn, not about
  keeping topics apart.
- [Anthropic's "orchestrator-workers" pattern](https://www.anthropic.com/research/building-effective-agents)
  is also different: a single orchestrator decomposes *one* task into
  subtasks dispatched to workers and synthesized back — a single-turn
  fan-out/fan-in, not a long-running, sticky, multi-turn router.

**Academic:**

- [Conversation Tree Architecture](https://arxiv.org/html/2603.21278v1)
  proposes nearly this exact idea conceptually: a tree of context-isolated
  topic nodes, to avoid "logical context poisoning" — topically distinct
  threads bleeding into one another in one flat window. Their own prototype
  reports **no empirical results yet** — the problem is validated as real
  and named, the fix is not yet proven.
- [Context Branching for LLM Conversations (ContextBranch)](https://arxiv.org/abs/2512.13914)
  reports the strongest *quantitative* evidence found for the underlying
  hypothesis: isolating unrelated topics into separate branches gave
  **58.1% context size reduction**, **+2.5%** overall response quality, and
  **+6.8%** "context awareness," vs. a flat linear conversation. Different
  specific mechanism (manual branch/checkpoint/merge for exploratory
  programming, not automatic topic routing), but real evidence that
  isolation-by-topic measurably helps.
- [SWRouter](https://arxiv.org/html/2609.11414) solves a related but
  different problem (routing between different LLM *models* per query), but
  its topic-shift detector — similarity-based window partitioning using an
  embedding-similarity threshold between adjacent turns — is a concrete,
  evaluated *alternative* to an LLM-call-based classifier for "is this the
  same topic." It reports **16.26%** improvement over the best single model
  and **8.22%** over a naive conversation-history baseline, at **~1.23x**
  the token cost — a real, quantified example of the router's own
  cost/benefit tradeoff.
- Decades of production customer-service chatbot routing (intent
  classification → domain-specific bot) confirm the two dominant failure
  modes this kind of system will hit in practice: out-of-domain/
  misclassified intent, and fragmented experience when a user's need
  genuinely spans two specialist domains at once.
