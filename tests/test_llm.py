"""DeepSeekClient.complete()'s retry-on-truncated-empty behavior.

Constructs a real DeepSeekClient (needs no network -- __init__ only builds
an `openai.OpenAI` object, it doesn't call anything) with a dummy key, then
replaces its inner `_client.chat.completions.create` with a scripted stand-in
so no real HTTP call is ever made.
"""

from hydra.llm import DeepSeekClient


class _Usage(object):
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Message(object):
    def __init__(self, content):
        self.content = content


class _Choice(object):
    def __init__(self, content, finish_reason):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response(object):
    def __init__(self, content, finish_reason, usage=None):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = usage or _Usage()


class _ScriptedCompletions(object):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _client_with_scripted_responses(responses):
    client = DeepSeekClient(api_key="dummy-key-for-tests")
    client._client.chat.completions = _ScriptedCompletions(responses)
    return client


def test_a_default_timeout_is_set_so_a_hung_request_cannot_block_forever():
    client = DeepSeekClient(api_key="dummy-key-for-tests")
    assert client.timeout == 120
    assert client._client.timeout is not None


def test_an_explicit_timeout_is_honored():
    client = DeepSeekClient(api_key="dummy-key-for-tests", timeout=5)
    assert client.timeout == 5


def test_a_clean_short_answer_needs_no_retry():
    client = _client_with_scripted_responses([_Response("EXISTING: t1", "stop")])
    result = client.complete(model="m", system="s", messages=[], max_tokens=60)
    assert result.text == "EXISTING: t1"
    assert len(client._client.chat.completions.calls) == 1


def test_truncated_empty_reply_retries_with_a_bigger_budget():
    client = _client_with_scripted_responses([
        _Response("", "length"),          # reasoning ate the whole budget
        _Response("EXISTING: t1", "stop"),  # succeeds once given more room
    ])
    result = client.complete(model="m", system="s", messages=[], max_tokens=60)
    assert result.text == "EXISTING: t1"
    calls = client._client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 60
    assert calls[1]["max_tokens"] == 180  # 3x the first attempt


def test_a_short_valid_answer_that_happens_to_hit_the_length_limit_is_not_retried():
    """finish_reason="length" with NON-empty text is a legitimately truncated
    (but present) answer, not the "reasoning ate everything" failure mode --
    retrying would waste a call for no reason since there IS a result.
    """
    client = _client_with_scripted_responses([_Response("EXISTING: t1 and also", "length")])
    result = client.complete(model="m", system="s", messages=[], max_tokens=60)
    assert result.text == "EXISTING: t1 and also"
    assert len(client._client.chat.completions.calls) == 1


def test_retries_are_capped_and_the_last_empty_result_is_returned_honestly():
    client = _client_with_scripted_responses([
        _Response("", "length"),
        _Response("", "length"),
        _Response("", "length"),
    ])
    result = client.complete(model="m", system="s", messages=[], max_tokens=60, max_retries=2)
    assert result.text == ""
    assert result.finish_reason == "length"
    calls = client._client.chat.completions.calls
    assert len(calls) == 3  # 1 initial + 2 retries, then give up honestly
    assert [c["max_tokens"] for c in calls] == [60, 180, 540]
