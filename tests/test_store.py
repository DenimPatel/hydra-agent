"""JSON persistence round-trips for a shard and a session registry."""

import os

from hydra import store
from hydra.shard import Shard


def test_a_corrupt_registry_raises_a_clear_error_instead_of_json_traceback(tmp_path):
    import pytest

    session_dir = str(tmp_path / "session")
    os.makedirs(session_dir)
    with open(os.path.join(session_dir, "registry.json"), "w") as handle:
        handle.write("{not valid json")

    with pytest.raises(store.RegistryError) as excinfo:
        store.load_registry(session_dir)
    assert "registry.json" in str(excinfo.value)


def test_a_corrupt_shard_file_raises_a_clear_error(tmp_path):
    import pytest

    session_dir = str(tmp_path / "session")
    store.ensure_session_dir(session_dir)
    with open(os.path.join(session_dir, "shards", "t1.json"), "w") as handle:
        handle.write("nope")

    with pytest.raises(store.RegistryError):
        store.load_shard(session_dir, "t1")


def test_a_registry_of_the_wrong_shape_raises(tmp_path):
    import pytest

    session_dir = str(tmp_path / "session")
    os.makedirs(session_dir)
    with open(os.path.join(session_dir, "registry.json"), "w") as handle:
        handle.write('["not", "an", "object"]')

    with pytest.raises(store.RegistryError):
        store.load_registry(session_dir)


def test_writes_are_atomic_and_leave_no_temp_files_behind(tmp_path):
    session_dir = str(tmp_path / "session")
    store.save_shard(session_dir, Shard("t1", "weather"))
    store.save_registry(session_dir, turn=1, shard_ids_to_labels={"t1": "weather"})

    leftovers = [name for name in os.listdir(session_dir) if name.startswith(".tmp-")]
    assert leftovers == []
    shard_tmp = [n for n in os.listdir(os.path.join(session_dir, "shards"))
                 if n.startswith(".tmp-")]
    assert shard_tmp == []
    assert store.load_registry(session_dir)["turn"] == 1


def test_save_and_load_a_single_shard_round_trips(tmp_path):
    session_dir = str(tmp_path / "session")
    shard = Shard("t1", "weather", created_turn=2)
    shard.messages = [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "hello"}]
    shard.turns = 1
    shard.last_input_tokens = 12

    store.save_shard(session_dir, shard)
    loaded = store.load_shard(session_dir, "t1")

    assert loaded.shard_id == "t1"
    assert loaded.topic_label == "weather"
    assert loaded.messages == shard.messages
    assert loaded.turns == 1
    assert loaded.last_input_tokens == 12


def test_loading_a_missing_shard_returns_none(tmp_path):
    assert store.load_shard(str(tmp_path / "session"), "nope") is None


def test_registry_round_trips(tmp_path):
    session_dir = str(tmp_path / "session")
    store.save_registry(session_dir, turn=5, shard_ids_to_labels={"t1": "weather", "t2": "code"})
    registry = store.load_registry(session_dir)
    assert registry["turn"] == 5
    assert registry["shards"] == {"t1": "weather", "t2": "code"}


def test_load_all_shards_reconstructs_every_shard_from_the_registry(tmp_path):
    session_dir = str(tmp_path / "session")
    store.save_shard(session_dir, Shard("t1", "weather"))
    store.save_shard(session_dir, Shard("t2", "code"))
    store.save_registry(session_dir, turn=7, shard_ids_to_labels={"t1": "weather", "t2": "code"})

    shards, turn = store.load_all_shards(session_dir)
    assert turn == 7
    assert set(shards.keys()) == {"t1", "t2"}
    assert shards["t1"].topic_label == "weather"


def test_an_unwritten_session_dir_loads_as_empty(tmp_path):
    shards, turn = store.load_all_shards(str(tmp_path / "never-written"))
    assert shards == {}
    assert turn == 0
    assert not os.path.exists(str(tmp_path / "never-written"))
