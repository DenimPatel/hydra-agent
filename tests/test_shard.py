"""Shard.send() bookkeeping and to_dict/from_dict round-tripping."""

from hydra.shard import Shard

from .fakes import FakeLLMClient, text


def test_send_appends_both_turns_and_updates_bookkeeping():
    client = FakeLLMClient([text("it's sunny", input_tokens=20, output_tokens=4)])
    shard = Shard("t1", "weather")
    reply = shard.send(client, "fake-model", "what's the weather?")
    assert reply == "it's sunny"
    assert shard.messages == [
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": "it's sunny"},
    ]
    assert shard.turns == 1
    assert shard.context_tokens() == 20


def test_a_shard_never_sees_another_shards_messages():
    client = FakeLLMClient([text("reply a"), text("reply b")])
    a = Shard("t1", "weather")
    b = Shard("t2", "code")
    a.send(client, "fake-model", "hi a")
    b.send(client, "fake-model", "hi b")
    assert client.calls[0]["messages"] == [{"role": "user", "content": "hi a"}]
    assert client.calls[1]["messages"] == [{"role": "user", "content": "hi b"}]


def test_a_failed_send_leaves_history_and_bookkeeping_untouched():
    """A raised call must not append the user turn: otherwise the dangling
    message survives, gets persisted, and is resent forever with no reply.
    """
    client = FakeLLMClient([RuntimeError("network down")])
    shard = Shard("t1", "weather")
    shard.messages = [{"role": "user", "content": "earlier"},
                       {"role": "assistant", "content": "earlier reply"}]
    before = list(shard.messages)

    try:
        shard.send(client, "fake-model", "the failing one")
    except RuntimeError:
        pass

    assert shard.messages == before
    assert shard.turns == 0
    assert shard.last_input_tokens == 0


def test_a_retry_after_a_failure_sends_only_the_new_message(tmp_path):
    client = FakeLLMClient([RuntimeError("network down"), text("ok")])
    shard = Shard("t1", "weather")
    try:
        shard.send(client, "fake-model", "the failing one")
    except RuntimeError:
        pass
    shard.send(client, "fake-model", "the good one")
    assert client.calls[1]["messages"] == [{"role": "user", "content": "the good one"}]


def test_to_dict_from_dict_round_trips_everything():
    client = FakeLLMClient([text("sunny")])
    shard = Shard("t1", "weather", created_turn=3)
    shard.send(client, "fake-model", "hi")
    restored = Shard.from_dict(shard.to_dict())
    assert restored.shard_id == shard.shard_id
    assert restored.topic_label == shard.topic_label
    assert restored.messages == shard.messages
    assert restored.turns == shard.turns
    assert restored.last_input_tokens == shard.last_input_tokens
    assert restored.created_turn == shard.created_turn
