"""A tmp qwn home with a small corpus: a 2-page text PDF, a PNG and a markdown file."""

from pathlib import Path

import pytest
from fakes import FakeReranker
from sample_corpus import CountingEmbedder, make_corpus

from qwn.config import Settings
from qwn.ingest import Ingester, open_index
from qwn.models import Registry


@pytest.fixture
def settings(home) -> Settings:
    return Settings()


@pytest.fixture
def embedder(settings) -> CountingEmbedder:
    return CountingEmbedder(settings.embed_dim)


@pytest.fixture
def registry(settings, embedder) -> Registry:
    return Registry(settings, overrides={"embedder": embedder, "reranker": FakeReranker()})


@pytest.fixture
def corpus(home) -> Path:
    return make_corpus(home)


@pytest.fixture
def index(settings):
    return open_index(settings)


@pytest.fixture
def ingester(settings, index, registry) -> Ingester:
    return Ingester(settings, index, registry.embedder)
