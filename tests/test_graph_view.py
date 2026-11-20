"""The live graph diagram: drawn from the compiled graph, coloured by the
ProgressTracker's counters."""
import dataclasses
from collections import Counter

import pytest

from agent.components import AgentComponents
from agent.config import AgentConfig
from agent.graph import build_graph
from agent.graph_view import IDLE, RUNNING, VISITED, graph_dot
from agent.progress import ProgressTracker
from test_progress import _run
from test_graph import PASS, REPOCODER, RLCODER, FakeComponents, _unsupported


@pytest.fixture(scope="module")
def drawable():
    nothing = AgentComponents(**{f.name: None for f in dataclasses.fields(AgentComponents)})
    return build_graph(AgentConfig(), components=nothing).get_graph(xray=True)


def _node(dot: str, node_id: str) -> str:
    return next(line for line in dot.splitlines() if line.strip().startswith(f'"{node_id}" ['))


def _has(line: str, style: dict) -> bool:
    return all(f'{k}="{v}"' in line for k, v in style.items())


def test_every_node_of_the_compiled_graph_is_drawn_and_the_subgraph_is_a_cluster(drawable):
    dot = graph_dot(drawable)
    for node_id in drawable.nodes:
        if node_id.startswith("corpus_search:__"):
            assert f'"{node_id}"' not in dot  # its own start / end are left out
        else:
            assert _has(_node(dot, node_id), IDLE)
    assert 'subgraph "cluster_corpus_search"' in dot
    # Edges into the subgraph go to its first real node; conditional ones are dashed.
    assert '"decompose" -> "corpus_search:retrieve_corpus" [style="dashed"];' in dot
    assert '"generate" -> "reflect" ;' in dot


def test_running_visited_and_repeat_counts(drawable):
    dot = graph_dot(drawable, visited=Counter(router=1, generate=2, retrieve_corpus=3), running=Counter(reflect=1))
    assert _has(_node(dot, "reflect"), RUNNING)
    assert _has(_node(dot, "router"), VISITED)
    assert 'label="generate ×2"' in _node(dot, "generate")
    assert 'label="retrieve_corpus ×3"' in _node(dot, "corpus_search:retrieve_corpus")
    assert _has(_node(dot, "decline"), IDLE)
    assert _has(_node(dot, "__start__"), VISITED)
    assert _has(_node(dot, "__end__"), IDLE)  # not finished yet


def test_a_finished_run_colours_its_path(drawable):
    fake = FakeComponents(corpus={"q": [REPOCODER, RLCODER]}, critiques=[_unsupported(), PASS])
    tracker, _ = _run(fake)
    assert not tracker.running  # everything that started finished
    assert tracker.visited["generate"] == 2 and tracker.visited["reflect"] == 2

    dot = graph_dot(drawable, tracker.visited)
    for node_id in ("contextualize", "router", "decompose", "corpus_search:grade_docs", "merge_results", "finalize", "__end__"):
        assert _has(_node(dot, node_id), VISITED), node_id
    for node_id in ("decline", "web_search", "fallback_research", "search_aspects", "corpus_search:rewrite_query"):
        assert _has(_node(dot, node_id), IDLE), node_id


def test_tracker_counts_parallel_runs_while_they_overlap():
    tracker = ProgressTracker()
    tracker.update({"id": "a", "name": "grade_docs", "input": {}})
    tracker.update({"id": "b", "name": "grade_docs", "input": {}})
    assert tracker.running["grade_docs"] == 2
    tracker.update({"id": "a", "name": "grade_docs", "result": {}})
    assert tracker.running["grade_docs"] == 1 and tracker.visited["grade_docs"] == 1
    tracker.update({"id": "b", "name": "grade_docs", "result": {}})
    assert "grade_docs" not in tracker.running and tracker.visited["grade_docs"] == 2
