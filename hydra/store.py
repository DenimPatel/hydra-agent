"""On-disk persistence for a session's registry and shards, as plain JSON --
no database, no append-only ledger. A session directory looks like:

    <session_dir>/
      registry.json        -- {"turn": N, "shards": {shard_id: topic_label}}
      shards/
        <shard_id>.json     -- one Shard.to_dict()

Loading a session is just reading these files back into objects; there is
no separate replay step, because the messages list IS the state.

Writes are atomic (temp file + `os.replace`): a crash mid-write can never
leave a torn `registry.json`/shard file behind. `RegistryError` is raised
when a session file exists but cannot be read or parsed, so the CLI can
report one clear line instead of dumping a raw JSON traceback at startup.
"""

import json
import os
import tempfile

from .shard import Shard

REGISTRY_FILENAME = "registry.json"


class RegistryError(Exception):
    """A session file exists but cannot be read or parsed."""


def ensure_session_dir(session_dir):
    os.makedirs(os.path.join(session_dir, "shards"), exist_ok=True)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as exc:  # includes json.JSONDecodeError
        raise RegistryError("could not parse %s: %s" % (path, exc)) from exc
    except OSError as exc:
        raise RegistryError("could not read %s: %s" % (path, exc)) from exc


def _write_json(path, data):
    """Atomic write: dump to a temp file in the same directory, then
    `os.replace` it into place (atomic on POSIX and Windows), so readers only
    ever see either the old complete file or the new complete file.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_registry(session_dir):
    path = os.path.join(session_dir, REGISTRY_FILENAME)
    if not os.path.exists(path):
        return {"turn": 0, "shards": {}}
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("shards"), dict):
        raise RegistryError(
            'malformed registry %s: expected {"turn": N, "shards": {...}}' % path)
    return data


def save_registry(session_dir, turn, shard_ids_to_labels):
    ensure_session_dir(session_dir)
    path = os.path.join(session_dir, REGISTRY_FILENAME)
    _write_json(path, {"turn": turn, "shards": dict(shard_ids_to_labels)})


def load_shard(session_dir, shard_id):
    path = os.path.join(session_dir, "shards", "%s.json" % shard_id)
    if not os.path.exists(path):
        return None
    return Shard.from_dict(_read_json(path))


def save_shard(session_dir, shard):
    ensure_session_dir(session_dir)
    path = os.path.join(session_dir, "shards", "%s.json" % shard.shard_id)
    _write_json(path, shard.to_dict())


def load_all_shards(session_dir):
    registry = load_registry(session_dir)
    shards = {}
    for shard_id in registry.get("shards", {}):
        shard = load_shard(session_dir, shard_id)
        if shard is not None:
            shards[shard_id] = shard
    return shards, registry.get("turn", 0)
