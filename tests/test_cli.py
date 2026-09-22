"""Session.handle_message's verbose auto-status printing, and print_status's
extended (verbose) columns. Uses a `rich.Console(record=True)` to capture
what would have gone to the terminal, with a FakeLLMClient so no network
call is ever made.
"""

from rich.console import Console

from hydra.config import Config
from hydra.cli import Session, status_rows

from .fakes import FakeLLMClient, text


def _console():
    return Console(record=True, width=200)


def test_verbose_off_by_default_prints_no_status_table_after_a_turn(tmp_path):
    client = FakeLLMClient([text("hello")])
    config = Config(data_dir=str(tmp_path))
    session = Session(client, config)  # verbose defaults to False
    console = _console()

    session.handle_message(console, "hi")

    output = console.export_text()
    assert "hydra session" not in output


def test_verbose_true_prints_the_extended_status_table_after_every_turn(tmp_path):
    client = FakeLLMClient([text("hello")])
    config = Config(data_dir=str(tmp_path))
    session = Session(client, config, verbose=True)
    console = _console()

    session.handle_message(console, "hi")

    output = console.export_text()
    assert "hydra session" in output
    assert "verbose" in output          # the (verbose) title suffix
    assert "out tok (last)" in output   # an extended-only column
    assert "turns since active" in output


def test_the_verbose_toggle_flips_the_sessions_own_flag(tmp_path):
    client = FakeLLMClient([])
    config = Config(data_dir=str(tmp_path))
    session = Session(client, config, verbose=False)
    assert session.verbose is False
    session.verbose = not session.verbose
    assert session.verbose is True


def test_status_command_always_shows_extended_columns_even_when_not_verbose(tmp_path):
    """/status is an explicit request for detail -- it shouldn't need
    verbose mode to be on to show the full picture.
    """
    client = FakeLLMClient([text("hello")])
    config = Config(data_dir=str(tmp_path))
    session = Session(client, config, verbose=False)
    console = _console()
    session.handle_message(console, "hi")  # create a shard to report on

    status_console = _console()
    session.print_status(status_console, extended=True)
    output = status_console.export_text()
    assert "out tok (last)" in output
    assert "turns since active" in output


def test_extended_status_reports_real_output_tokens_and_turns_since_active(tmp_path):
    client = FakeLLMClient([
        text("weather reply", input_tokens=20, output_tokens=7),
        text("NEW: code", input_tokens=15, output_tokens=3),
        text("code reply", input_tokens=30, output_tokens=9),
    ])
    config = Config(data_dir=str(tmp_path))
    session = Session(client, config)
    console = _console()

    session.handle_message(console, "what's the weather?")   # t1, no router call (first ever)
    session.handle_message(console, "unrelated code question")  # spawns t2

    rows = status_rows(session.router.shards, session.router.turn, extended=True)
    # exact cells, in sorted-id order: [id, topic, turns, ctx tok, created turn,
    # out tok (last), turns since active] -- t1 created at turn 0 and last used
    # on turn 1, so with the global turn now at 2 it has 1 turn since active.
    assert rows == [
        ["t1", "what's the weather?", "1", "20", "0", "7", "1"],
        ["t2", "code", "1", "30", "1", "9", "0"],
    ]


def test_status_rows_compact_form_omits_the_extended_columns(tmp_path):
    client = FakeLLMClient([text("hi", input_tokens=11, output_tokens=3)])
    session = Session(client, Config(data_dir=str(tmp_path)))
    session.handle_message(_console(), "hello")

    rows = status_rows(session.router.shards, session.router.turn, extended=False)
    assert rows == [["t1", "hello", "1", "11", "0"]]


def _run_repl(monkeypatch, tmp_path, client, lines):
    import hydra.cli as cli

    monkeypatch.setattr(cli, "DeepSeekClient", lambda *a, **k: client)
    console = Console(record=True, width=200)
    monkeypatch.setattr(cli, "Console", lambda *a, **k: console)
    pending = iter(lines)
    monkeypatch.setattr(console, "input", lambda prompt="": next(pending))
    rc = cli.main(["chat", "--data-dir", str(tmp_path)])
    return rc, console.export_text()


def test_slash_commands_are_handled_locally_and_never_sent_to_the_model(tmp_path, monkeypatch):
    client = FakeLLMClient([])  # any model call would raise
    rc, out = _run_repl(monkeypatch, tmp_path, client,
                        ["/help", "/status", "/unknown", "/quit"])
    assert rc == 0
    assert client.calls == []
    assert "unknown command '/unknown'" in out


def test_bare_switch_and_new_print_usage_instead_of_unknown_command(tmp_path, monkeypatch):
    client = FakeLLMClient([])
    rc, out = _run_repl(monkeypatch, tmp_path, client, ["/switch", "/new", "/quit"])
    assert rc == 0
    assert "usage: /switch <id>" in out
    assert "usage: /new <label>" in out
    assert "unknown command" not in out
    assert client.calls == []


def test_a_plain_message_still_reaches_the_shard(tmp_path, monkeypatch):
    client = FakeLLMClient([text("hello back")])
    rc, _ = _run_repl(monkeypatch, tmp_path, client, ["hi there", "/quit"])
    assert rc == 0
    assert len(client.calls) == 1
    assert client.calls[0]["messages"] == [{"role": "user", "content": "hi there"}]


def test_switch_forces_the_next_message_into_the_named_shard(tmp_path, monkeypatch):
    client = FakeLLMClient([text("first reply"), text("forced reply")])
    rc, out = _run_repl(monkeypatch, tmp_path, client,
                        ["create first thread", "/switch t1", "again", "/quit"])
    assert rc == 0
    assert "next message forced to t1" in out
    # the forced turn is appended to t1's own history, alongside its first turn
    assert [m["content"] for m in client.calls[1]["messages"]] == [
        "create first thread", "first reply", "again"]


