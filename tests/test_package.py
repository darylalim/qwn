import qwn.cli


def test_cli_entry_point_is_a_typer_app() -> None:
    assert qwn.cli.app.info.help
