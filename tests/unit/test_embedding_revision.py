import sys
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from src.retrieval.models.embedder import EmbedderWrapper, _resolve_hf_revision


@pytest.mark.parametrize("selector", ["", "main", "release", "a" * 40])
def test_selected_revision_is_resolved_and_loaded_as_immutable_commit(monkeypatch, selector):
    commit = "b" * 40
    download = Mock(return_value=f"/cache/models--org--model/snapshots/{commit}/config.json")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model = SimpleNamespace(encode=Mock(return_value=[[0.1] * 1024]))
    factory = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory)
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    wrapper = EmbedderWrapper("org/model", "/cache", revision=selector)
    assert wrapper.space_descriptor()["space_id"] == ""

    wrapper.load()

    download.assert_called_once_with(
        "org/model", "config.json", cache_dir="/cache", revision=selector or None
    )
    assert factory.call_args.kwargs["revision"] == commit
    assert wrapper.revision == commit
    assert wrapper.space_descriptor()["space_id"]


def test_branch_moving_changes_space_identity(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: SimpleNamespace(
                encode=lambda *args, **kwargs: [[0.1] * 1024]
            ),
        ),
    )
    spaces = []
    for commit in ("a" * 40, "b" * 40):
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda *args, **kwargs: f"/cache/snapshots/{commit}/config.json",
        )
        wrapper = EmbedderWrapper("org/model", revision="main")
        wrapper.load()
        spaces.append(wrapper.space_descriptor()["space_id"])
    assert spaces[0] != spaces[1]


@pytest.mark.parametrize(
    "path", ["/cache/refs/main/config.json", "/cache/snapshots/main/config.json"]
)
def test_unproven_artifact_path_is_not_an_identity(monkeypatch, path):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: path)
    assert _resolve_hf_revision("org/model", "/cache", "main") == ""


def test_resolution_failure_leaves_loaded_model_unstamped(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", Mock(side_effect=OSError("offline")))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: SimpleNamespace(
                encode=lambda *args, **kwargs: [[0.1] * 1024]
            ),
        ),
    )
    wrapper = EmbedderWrapper("org/model", revision="main")
    wrapper.load()
    assert wrapper.is_loaded
    assert wrapper.revision == ""
    assert wrapper.space_descriptor()["space_id"] == ""
    assert wrapper.space_descriptor()["producer_id"] == ""
