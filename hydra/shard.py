"""A Shard: one isolated, topic-specific conversation.

A shard never sees any other shard's messages. Switching away from a topic
and back does not summarize or evict anything -- the shard's message list
is simply not sent to the model again until it is next routed to, at which
point it resumes exactly where it left off.
"""

import time

DEFAULT_SHARD_SYSTEM = (
    "You are a helpful assistant focused on one topic area for this "
    "conversation thread. Answer naturally; you do not need to mention "
    "that you are one of several specialized threads."
)


class Shard(object):
    def __init__(self, shard_id, topic_label, system_prompt=None,
                 created_turn=0, messages=None, turns=0,
                 last_input_tokens=0, last_output_tokens=0,
                 created_at=None, last_active_at=None):
        self.shard_id = shard_id
        self.topic_label = topic_label
        self.system_prompt = system_prompt or DEFAULT_SHARD_SYSTEM
        self.created_turn = created_turn
        self.messages = list(messages or [])
        self.turns = turns
        self.last_input_tokens = last_input_tokens
        self.last_output_tokens = last_output_tokens
        self.created_at = created_at or time.time()
        self.last_active_at = last_active_at or self.created_at

    def send(self, client, model, user_text, max_tokens=1024):
        """Append the user turn, call the model with this shard's FULL own
        history (and nothing from any other shard), append the reply, and
        update bookkeeping. Returns the reply text.

        The user turn is committed to `self.messages` only AFTER the call
        succeeds. If `complete()` raises, this shard's history is left exactly
        as it was -- no dangling user message with no reply for the next turn
        to send (and to persist) forever.
        """
        outgoing = self.messages + [{"role": "user", "content": user_text}]
        result = client.complete(
            model=model, system=self.system_prompt, messages=outgoing,
            max_tokens=max_tokens,
        )
        self.messages = outgoing
        self.messages.append({"role": "assistant", "content": result.text})
        self.turns += 1
        self.last_input_tokens = result.input_tokens
        self.last_output_tokens = result.output_tokens
        self.last_active_at = time.time()
        return result.text

    def context_tokens(self):
        """The real size of what was actually sent on the last call. Before
        any call has been made yet, this is 0 -- there is nothing sent yet.
        """
        return self.last_input_tokens

    def to_dict(self):
        return {
            "shard_id": self.shard_id,
            "topic_label": self.topic_label,
            "system_prompt": self.system_prompt,
            "created_turn": self.created_turn,
            "messages": self.messages,
            "turns": self.turns,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            shard_id=data["shard_id"],
            topic_label=data["topic_label"],
            system_prompt=data.get("system_prompt"),
            created_turn=data.get("created_turn", 0),
            messages=data.get("messages", []),
            turns=data.get("turns", 0),
            last_input_tokens=data.get("last_input_tokens", 0),
            last_output_tokens=data.get("last_output_tokens", 0),
            created_at=data.get("created_at"),
            last_active_at=data.get("last_active_at"),
        )
