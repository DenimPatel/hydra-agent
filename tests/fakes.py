"""A scripted LLMClient: no network, no API key, no spend. Mirrors
`hydra.llm.ChatResult`'s shape exactly so router.py/shard.py can't tell the
difference between this and a real DeepSeekClient.
"""

from hydra.llm import ChatResult


class FakeLLMClient(object):
    def __init__(self, replies):
        """`replies` is a list of ChatResult (or Exception) objects, or plain
        strings (wrapped into a ChatResult with token counts derived from
        length), popped in order, one per expected `.complete()` call.
        """
        self.replies = list(replies)
        self.calls = []

    def complete(self, model, system, messages, max_tokens=1024):
        self.calls.append({"model": model, "system": system, "messages": list(messages),
                            "max_tokens": max_tokens})
        if not self.replies:
            raise AssertionError("FakeLLMClient ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            return ChatResult(reply, input_tokens=len(reply) // 4, output_tokens=len(reply) // 4)
        return reply


def text(s, input_tokens=10, output_tokens=5):
    return ChatResult(s, input_tokens=input_tokens, output_tokens=output_tokens)
