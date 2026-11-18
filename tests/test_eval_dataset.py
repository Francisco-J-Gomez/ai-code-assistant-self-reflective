from collections import Counter

import pytest

from evals.dataset_io import CATEGORIES, load_examples, validate


def test_dataset_loads_and_is_valid():
    examples = load_examples()
    assert 30 <= len(examples) <= 50


def test_every_category_is_covered():
    counts = Counter(e["category"] for e in load_examples())
    assert set(counts) == CATEGORIES
    assert min(counts.values()) >= 5


def test_follow_ups_carry_chat_history():
    for e in load_examples():
        if e["category"] == "follow_up":
            assert e["inputs"]["chat_history"], e["id"]


def _example(**overrides):
    base = {
        "id": "x-01",
        "category": "corpus",
        "inputs": {"question": "q?", "chat_history": []},
        "outputs": {"expected_route": "corpus", "expected_arxiv_ids": ["1"], "reference": "r"},
    }
    base.update(overrides)
    return base


def test_validate_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        validate([_example(), _example()])


def test_validate_rejects_unknown_category():
    with pytest.raises(ValueError, match="category"):
        validate([_example(category="nope")])


def test_validate_rejects_decline_with_expected_papers():
    out = {"expected_route": "decline", "expected_arxiv_ids": ["1"], "reference": "r"}
    with pytest.raises(ValueError, match="decline"):
        validate([_example(outputs=out)])
