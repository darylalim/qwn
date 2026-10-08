from pathlib import Path

import pytest

from qwn.config import HomeNotFound, Settings, find_home


def test_defaults_resolve_relative_paths_against_home(home):
    s = Settings()
    assert s.home == home
    assert s.index_dir == home / "index"
    assert s.data_dir == home / "data"
    assert s.top_k == 50


def test_precedence_init_over_env_over_toml(home, monkeypatch):
    (home / "qwn.toml").write_text("top_k = 7\nrerank_k = 3\nfts_k = 9\n")
    monkeypatch.setenv("QWN_RERANK_K", "4")
    monkeypatch.setenv("QWN_FTS_K", "11")
    s = Settings(fts_k=13)
    assert s.top_k == 7  # toml over default
    assert s.rerank_k == 4  # env over toml
    assert s.fts_k == 13  # init kwargs over env


def test_absolute_paths_are_kept(home, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("idx")
    assert Settings(index_dir=elsewhere).index_dir == elsewhere


def test_unknown_toml_key_is_an_error(home):
    (home / "qwn.toml").write_text("top_kk = 7\n")
    with pytest.raises(ValueError, match="top_kk"):
        Settings()


def test_find_home_walks_up_to_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("QWN_HOME", raising=False)
    (tmp_path / "qwn.example.toml").write_text("")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_home(nested) == tmp_path


def test_find_home_without_marker_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("QWN_HOME", raising=False)
    with pytest.raises(HomeNotFound, match="QWN_HOME"):
        find_home(tmp_path)


def test_repo_has_a_home_marker():
    assert (Path(__file__).parents[2] / "qwn.example.toml").is_file()
