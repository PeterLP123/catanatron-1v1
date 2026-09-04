import pytest
from click.testing import CliRunner

from catanatron.cli.cli_players import CliPlayer, parse_cli_string
from catanatron.cli.play import simulate
from catanatron.models.player import Color


@pytest.mark.parametrize("spec", ["R,unknown", "R,", "R,R,R,R,R"])
def test_invalid_player_spec_fails_before_constructing_any_players(monkeypatch, spec):
    def unexpected_construction(*args):
        pytest.fail("Invalid configuration must fail before constructing players")

    monkeypatch.setattr(
        "catanatron.cli.cli_players.CLI_PLAYERS",
        [CliPlayer("R", "Test", "Test", unexpected_construction)],
    )
    with pytest.raises(ValueError):
        parse_cli_string(spec)


def test_player_specs_preserve_seat_order_and_parameters(monkeypatch):
    monkeypatch.setattr(
        "catanatron.cli.cli_players.CLI_PLAYERS",
        [CliPlayer("TEST", "Test", "Test", lambda *args: args)],
    )
    colors = list(Color)
    assert parse_cli_string(" TEST:2:True , TEST ") == [
        (colors[0], "2", "True"),
        (colors[1],),
    ]


@pytest.mark.parametrize(
    "args, message",
    [
        (["--players", "R,typo"], "Unknown player code"),
        (["--output", "unused"], "--output requires --output-format"),
    ],
)
def test_invalid_cli_arguments_return_usage_error(args, message):
    result = CliRunner().invoke(simulate, args)
    assert result.exit_code == 2
    assert message in result.output
