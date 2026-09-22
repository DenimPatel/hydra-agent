"""hydra-agent: routes each message to a topic-specific, isolated LLM shard
instead of growing one context and evicting from it.

See docs/design/orchestration-idea.md for the why, and
docs/architecture.md for how the pieces here fit together.
"""

__version__ = "0.1.0"
