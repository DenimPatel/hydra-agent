"""A minimal, standalone LLM client for DeepSeek's OpenAI-compatible API.

Deliberately independent of any other codebase: this project must not import
or depend on anything outside this repo. `DeepSeekClient` is a thin wrapper
translating DeepSeek's OpenAI-shaped chat-completions response into the
small `ChatResult` shape the rest of hydra uses -- so `router.py`/`shard.py`
only ever depend on that shape, never on the `openai` package's own types or
on DeepSeek/OpenAI-specific field names.

`FakeLLMClient` (in tests/fakes.py) implements the same duck-typed
`.complete(system, messages)` interface, so router/shard logic is fully
testable with zero network calls.
"""

import os


class ChatResult(object):
    """What every LLMClient.complete() call returns, provider-agnostic.

    `finish_reason` matters more here than in most callers: a REASONING
    model (deepseek-flash included) spends completion tokens on hidden
    reasoning before writing anything visible, so a too-small `max_tokens`
    truncates to an EMPTY `text` with `finish_reason="length"` -- not an
    error, not a short-but-valid answer. Callers that treat an empty reply
    as just "unparseable" lose this distinction; surfacing it here lets
    router.py say so explicitly instead of guessing.
    """

    def __init__(self, text, input_tokens=0, output_tokens=0, finish_reason=None):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason


class DeepSeekClient(object):
    """Wraps DeepSeek's OpenAI-compatible /chat/completions endpoint.

    DeepSeek's API is OpenAI-compatible but not Anthropic-shaped: no
    `tool_use`/`tool_result` blocks are used anywhere in hydra (the router
    and shards are both plain text-in/text-out), so a bare
    `chat.completions.create(messages=[{role, content}, ...])` call is all
    that's needed here -- no translation layer beyond reading `.usage` back
    into a `ChatResult`.
    """

    def __init__(self, api_key=None, base_url=None, timeout=None):
        import openai  # imported lazily so tests never need it installed

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set -- export it before running hydra, "
                "e.g. `export DEEPSEEK_API_KEY=sk-...`"
            )
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if timeout is None:
            timeout = float(os.environ.get("DEEPSEEK_TIMEOUT", "120"))
        self.timeout = timeout
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url,
                                     timeout=self.timeout)

    def complete(self, model, system, messages, max_tokens=1024, max_retries=2):
        """`messages` is a list of {"role": "user"|"assistant", "content": str}.
        `system` is prepended as a system-role message, matching how DeepSeek's
        (and every OpenAI-compatible) chat API expects it.

        RETRY ON TRUNCATED-EMPTY, NOT JUST A BIGGER STATIC BUDGET: measured
        directly against deepseek-flash, the hidden reasoning length for the
        SAME prompt varies enormously call to call -- 130 reasoning tokens on
        one attempt, 400+ (hitting a 400 cap with finish_reason="length" and
        empty visible text) on another, for the identical menu and message.
        No single fixed `max_tokens` is safe: raising it only lowers the
        failure rate, it doesn't remove it, since reasoning length here is
        unbounded on the high end for a sufficiently ambiguous decision. So
        instead of guessing a bigger constant, retry with a larger budget
        specifically when a response comes back both EMPTY and
        finish_reason="length" (the unambiguous signature of "reasoning ate
        the whole budget," not a legitimately short-but-valid answer) --
        each retry triples the budget, up to `max_retries` extra attempts.
        """
        full_messages = [{"role": "system", "content": system}] + list(messages)
        budget = max_tokens
        response = None
        for attempt in range(max_retries + 1):
            response = self._client.chat.completions.create(
                model=model, messages=full_messages, max_tokens=budget,
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            if text or finish_reason != "length" or attempt == max_retries:
                break
            budget *= 3  # reasoning ate the whole budget last time -- give it much more room

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        return ChatResult(text, input_tokens=input_tokens, output_tokens=output_tokens,
                           finish_reason=finish_reason)
