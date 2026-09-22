"""Router.classify/route: EXISTING/NEW parsing, the never-guess whitelist
check, and the "every failure falls back to NEW" discipline.
"""

from hydra.config import Config
from hydra.llm import ChatResult
from hydra.router import Router
from hydra.shard import Shard

from .fakes import FakeLLMClient, text


def _config():
    return Config(router_model="fake-router", shard_model="fake-shard")


def test_first_message_ever_is_always_new_with_no_call_made():
    client = FakeLLMClient([])  # any call would raise
    router = Router(client, _config())
    shard, decision = router.route("what's the weather like?")
    assert decision.is_new
    assert shard.topic_label
    assert client.calls == []


def test_a_second_unrelated_topic_spawns_a_new_shard():
    client = FakeLLMClient([text("NEW: code review")])
    router = Router(client, _config())
    weather_shard, _ = router.route("what's the weather like?")
    code_shard, decision = router.route("can you review this function?")
    assert decision.is_new
    assert code_shard.shard_id != weather_shard.shard_id
    assert code_shard.topic_label == "code review"


def test_returning_to_a_dormant_topic_routes_back_to_the_same_shard():
    client = FakeLLMClient([
        text("NEW: code"),
        text("EXISTING: t1"),
    ])
    router = Router(client, _config())
    weather_shard, _ = router.route("what's the weather?")
    router.route("let's talk code instead")  # spawns t2, consumes the NEW reply
    same_weather_shard, decision = router.route("back to the weather question")
    assert not decision.is_new
    assert same_weather_shard.shard_id == weather_shard.shard_id


def test_a_hallucinated_shard_id_is_never_trusted_falls_back_to_new():
    client = FakeLLMClient([text("EXISTING: t99")])
    router = Router(client, _config(), shards={"t1": Shard("t1", "weather")})
    shard, decision = router.route("something")
    assert decision.is_new
    assert "never guessed" in decision.reason
    assert shard.shard_id != "t1"


def test_an_unparseable_reply_falls_back_to_new():
    client = FakeLLMClient([text("I'm not sure what you mean")])
    router = Router(client, _config(), shards={"t1": Shard("t1", "weather")})
    shard, decision = router.route("something")
    assert decision.is_new
    assert "unparseable" in decision.reason


def test_a_raising_client_falls_back_to_new():
    client = FakeLLMClient([RuntimeError("network down")])
    router = Router(client, _config(), shards={"t1": Shard("t1", "weather")})
    shard, decision = router.route("something")
    assert decision.is_new
    assert "router call failed" in decision.reason


def test_a_reasoning_model_truncated_to_empty_by_max_tokens_falls_back_to_new_with_a_clear_reason():
    """A real, observed failure mode: a reasoning model (deepseek-flash) can
    spend its entire completion budget on hidden reasoning, returning an
    EMPTY visible reply with finish_reason="length" -- not a crash, not a
    garbled answer. This must still fall back to NEW (never guess), but the
    reason should name the actual cause, not just say "unparseable".
    """
    empty_truncated = ChatResult("", input_tokens=59, output_tokens=60, finish_reason="length")
    client = FakeLLMClient([empty_truncated])
    router = Router(client, _config(), shards={"t1": Shard("t1", "weather")})
    shard, decision = router.route("something")
    assert decision.is_new
    assert "max_tokens" in decision.reason
    assert "reasoning" in decision.reason


def test_the_router_call_reports_its_own_cost():
    client = FakeLLMClient([text("NEW: weather", input_tokens=40, output_tokens=6)])
    router = Router(client, _config(), shards={"t1": Shard("t1", "code")})
    _, decision = router.route("what's the weather")
    assert decision.cost_tokens == 46


def test_a_gap_in_loaded_ids_never_reuses_an_existing_id():
    """A missing/corrupt shard file leaves non-contiguous ids on disk. The old
    `len(shards) + 1` would hand out "t3" here and overwrite the loaded t3 --
    silently destroying it. The next id must come from the max, not the count.
    """
    existing = {"t1": Shard("t1", "weather"), "t3": Shard("t3", "code")}
    router = Router(FakeLLMClient([]), _config(), shards=existing)
    spawned = router._spawn_shard("new topic")
    assert spawned.shard_id == "t4"
    assert router.shards["t3"] is existing["t3"]  # untouched


def test_public_spawn_creates_a_shard_and_advances_the_turn():
    router = Router(FakeLLMClient([]), _config(), turn=5)
    shard = router.spawn("weather")
    assert shard.shard_id == "t1"
    assert shard.created_turn == 5   # snapshot before the increment
    assert router.turn == 6
    assert router.shards == {"t1": shard}
