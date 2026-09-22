"""Every tunable in one place."""

import os


class Config(object):
    def __init__(
        self,
        router_model="deepseek-flash",
        shard_model="deepseek-flash",
        data_dir=".hydra",
        session_name="default",
        router_max_tokens=400,     # the VISIBLE reply is one short line, but a
                                   # reasoning model (e.g. deepseek-flash) spends
                                   # completion tokens on hidden reasoning FIRST --
                                   # too small a budget here truncates to an EMPTY
                                   # visible answer (finish_reason "length") before
                                   # a single word of the actual verdict is written.
                                   # Seen directly: max_tokens=60 produced an empty
                                   # reply 100% of the time in manual testing.
        shard_max_tokens=1024,
        router_system_prompt=None,  # None -> use router.DEFAULT_ROUTER_SYSTEM
        shard_system_prompt=None,   # None -> use shard.DEFAULT_SHARD_SYSTEM
    ):
        self.router_model = router_model
        self.shard_model = shard_model
        self.data_dir = data_dir
        self.session_name = session_name
        self.router_max_tokens = router_max_tokens
        self.shard_max_tokens = shard_max_tokens
        self.router_system_prompt = router_system_prompt
        self.shard_system_prompt = shard_system_prompt

    def session_dir(self):
        return os.path.join(self.data_dir, "sessions", self.session_name)
