import subprocess
import sys
import textwrap

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


def test_cli_baselines_and_help_work_without_optional_dependencies(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
            import importlib.abc
            import sys

            optional = {
                'numpy', 'pandas', 'gymnasium', 'torch', 'stable_baselines3',
                'sb3_contrib', 'textual', 'pyarrow', 'fastparquet', 'tensorboard'
            }

            class CoreOnly(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in optional:
                        raise ModuleNotFoundError(fullname, name=fullname)

            sys.meta_path.insert(0, CoreOnly())
            from click.testing import CliRunner
            from catanatron.cli.play import simulate
            runner = CliRunner()
            for args in (
                ['--help'], ['--help-players'],
                ['--colonist-1v1', '--players', 'F,R', '--num', '1', '--seed', '7']
            ):
                result = runner.invoke(simulate, args)
                assert result.exit_code == 0, (result.output, result.exception)
            for code in ('L', 'T', 'C', 'O', 'N', 'Q'):
                result = runner.invoke(simulate, ['--players', f'{code}:missing,R'])
                assert result.exit_code == 2, (code, result.exception)
                assert 'catanatron-1v1[gym,colonist]' in result.output
            assert not optional.intersection(sys.modules)
        """
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
