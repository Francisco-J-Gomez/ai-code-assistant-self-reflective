"""The chat UI's progress log, fed with the real `tasks` stream of the
graph (fake components, no network)."""
from langchain_core.messages import HumanMessage

from agent.config import AgentConfig
from agent.progress import ProgressTracker
from test_graph import OFF_TOPIC, PASS, REPOCODER, RLCODER, FakeComponents, _graph, _unsupported

PART_1, PART_2 = "RAG for code completion", "LLMs for test generation"


def _run(fake, question="q", thread_id="p", graph=None):
    graph = graph or _graph(fake)
    tracker = ProgressTracker(AgentConfig(max_iterations=3))
    labels = []
    for _ns, event in graph.stream(
        {"messages": [HumanMessage(question)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="tasks",
        subgraphs=True,
    ):
        tracker.update(event)
        if not labels or labels[-1] != tracker.current:
            labels.append(tracker.current)
    return tracker, labels


def test_full_corrective_path_with_a_revision():
    fake = FakeComponents(
        sub_queries=[PART_1, PART_2],
        corpus={PART_1: [REPOCODER, RLCODER, OFF_TOPIC]},
        critiques=[_unsupported(), PASS],
    )
    tracker, labels = _run(fake)

    steps = tracker.steps
    assert steps[0] == "📄 Research question → searching the paper corpus"
    assert steps[1] == f"Split into 2 parts: 1) “{PART_1}” 2) “{PART_2}”"
    # Parallel parts are labelled, so their interleaved lines stay readable.
    assert "Part 1: found 3 new papers" in steps
    assert "Part 1: 2 of 3 relevant" in steps
    assert "Part 2: 0 of 1 relevant" in steps
    assert f"Part 2: too few relevant papers → new query “rewritten: {PART_2}”" in steps
    assert f"Not covered by the corpus: “{PART_2}”" in steps
    assert "🔎 Live search found 2 sources" in steps
    assert steps[-4:] == [
        "Draft written",
        "⚠️ Fact-check: 1 unsupported claim → revising",
        "Draft 2 written",
        "✅ Fact-check passed: 1 of 1 claims supported",
    ]
    assert "Revising the answer (draft 2 of 3)…" in labels
    assert labels[-1] == "Finishing…"


def test_single_part_question_has_no_part_labels():
    tracker, _ = _run(FakeComponents(corpus={"q": [REPOCODER, RLCODER]}))
    assert "Found 2 new papers" in tracker.steps
    assert "2 of 2 relevant" in tracker.steps
    assert not any(s.startswith("Part ") for s in tracker.steps)
    assert "2 relevant papers to answer from" in tracker.steps


def test_web_route_and_last_draft_kept():
    fake = FakeComponents(route="web", critiques=[_unsupported()] * 3)
    tracker, _ = _run(fake)

    assert tracker.steps[:2] == ["🌐 Tool / library question → searching the web", "🌐 Found 1 web result"]
    assert tracker.steps[-1] == "⚠️ Fact-check: 1 unsupported claim → keeping the best draft"


def test_decline_and_follow_up():
    fake = FakeComponents(standalone="What benchmark did RepoCoder introduce?")
    graph = _graph(fake)
    _run(fake, "What is RepoCoder?", "f", graph)
    tracker, _ = _run(fake, "What benchmark did they introduce?", "f", graph)
    assert tracker.steps[0] == "Follow-up understood as “What benchmark did RepoCoder introduce?”"

    tracker, _ = _run(FakeComponents(route="out_of_scope"))
    assert tracker.steps == ["Outside AI / software development → declining"]
