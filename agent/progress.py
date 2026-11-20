"""Human-readable progress for a graph run, for the chat UI.

Consumes LangGraph's `tasks` stream (`graph.stream(..., stream_mode="tasks",
subgraphs=True)`): one event when a node starts (with its input state) and
one when it finishes (with its update). From these it keeps

    current  what the agent is doing right now ("Fact-checking the answer...")
    steps    one line per finished step with its outcome ("3 of 5 papers
             relevant", "Fact-check: 1 unsupported claim -> revising")
    running  node name -> how many runs of it are in progress (parallel
             corpus_search runs can overlap)
    visited  node name -> how many times it finished, for the live graph
             diagram (agent/graph_view.py)

Pure (no Streamlit), so it is unit-tested; ui/chat_app.py only renders it.
Steps with nothing worth saying (e.g. report, finalize) produce no line.
"""
from collections import Counter
from typing import Any

from agent.config import AgentConfig


def _get(obj: Any, key: str, default=None):
    """Node inputs are pydantic states (or a plain dict for a Send)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):  # [(channel, value), ...]
        return {k: v for k, v in result if isinstance(k, str)}
    return {}


def _quote(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return f"“{text if len(text) <= limit else text[: limit - 1] + '…'}”"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


class ProgressTracker:
    def __init__(self, config: AgentConfig | None = None):
        self.max_iterations = (config or AgentConfig()).max_iterations
        self.current = "Starting…"
        self.steps: list[str] = []
        self._inputs: dict[str, Any] = {}  # task id -> input, to describe its result
        self._parts = 1
        self.running: Counter[str] = Counter()
        self.visited: Counter[str] = Counter()

    # -- public -------------------------------------------------------------

    def update(self, event: dict) -> str | None:
        """Feed one `tasks` stream event. Returns the new step line, if the
        event finished a step worth reporting (also appended to `steps`)."""
        name, task_id = event.get("name", ""), event.get("id", "")
        if "result" not in event:  # task started
            self.running[name] += 1
            self._inputs[task_id] = event.get("input")
            label = self._started(name, event.get("input"))
            if label:
                self.current = label
            return None
        self.running[name] -= 1
        if self.running[name] <= 0:
            del self.running[name]
        self.visited[name] += 1
        line = self._finished(name, self._inputs.pop(task_id, None), _as_dict(event.get("result")))
        if line:
            line = line[0].upper() + line[1:]
            self.steps.append(line)
        return line

    # -- descriptions ---------------------------------------------------------

    def _part(self, state: Any) -> str:
        """'Part 2: ' when the question was split, '' otherwise."""
        index = _get(state, "index")
        return f"Part {index + 1}: " if self._parts > 1 and index is not None else ""

    def _started(self, name: str, state: Any) -> str | None:
        part = self._part(state)
        iteration = _get(state, "iteration", 0) or 0
        labels = {
            "contextualize": "Reading the question…",
            "router": "Choosing where to look…",
            "decompose": "Breaking the question into parts…",
            "retrieve_corpus": f"{part}searching the paper corpus…",
            "grade_docs": f"{part}checking which papers are relevant…",
            "rewrite_query": f"{part}rewriting the search query…",
            "merge_results": "Combining what was found…",
            "fallback_research": "Searching the web and live arXiv…",
            "web_search": "Searching the web…",
            "generate": (
                f"Revising the answer (draft {iteration + 1} of {self.max_iterations})…" if iteration else "Writing the answer…"
            ),
            "reflect": "Fact-checking the answer against its sources…",
            "search_aspects": "Searching for the parts the answer missed…",
            "finalize": "Finishing…",
        }
        label = labels.get(name)
        return label[0].upper() + label[1:] if label else None

    def _finished(self, name: str, state: Any, result: dict) -> str | None:
        part = self._part(state)

        if name == "contextualize":
            question, standalone = result.get("question"), result.get("standalone_question")
            if standalone and question and standalone != question:
                return f"Follow-up understood as {_quote(standalone)}"
            return None

        if name == "router":
            return {
                "corpus": "📄 Research question → searching the paper corpus",
                "web": "🌐 Tool / library question → searching the web",
                "decline": "Outside AI / software development → declining",
            }.get(result.get("route"))

        if name == "decompose":
            sub_queries = result.get("sub_queries") or []
            self._parts = max(len(sub_queries), 1)
            if len(sub_queries) > 1:
                parts = " ".join(f"{i}) {_quote(q, 70)}" for i, q in enumerate(sub_queries, 1))
                return f"Split into {len(sub_queries)} parts: {parts}"
            return None

        if name == "retrieve_corpus":
            n = len(result.get("candidates") or [])
            return f"{part}found {_plural(n, 'new paper')}" if n else f"{part}no new papers found"

        if name == "grade_docs":
            checked = len(_get(state, "candidates", []) or [])
            if not checked:
                return None
            kept_before = len(_get(state, "documents", []) or [])
            relevant = len(result.get("documents") or []) - kept_before
            return f"{part}{relevant} of {checked} relevant"

        if name == "rewrite_query":
            return f"{part}too few relevant papers → new query {_quote(result.get('search_query', ''))}"

        if name == "merge_results":
            missing = result.get("missing") or []
            if missing:
                return "Not covered by the corpus: " + "; ".join(_quote(m, 70) for m in missing)
            n = len(result.get("documents") or [])
            return f"{_plural(n, 'relevant paper')} to answer from"

        if name == "fallback_research":
            n = len(result.get("documents") or []) - len(_get(state, "documents", []) or [])
            return f"🔎 Live search found {_plural(n, 'source')}"

        if name == "web_search":
            return f"🌐 Found {_plural(len(result.get('documents') or []), 'web result')}"

        if name == "generate":
            iteration = result.get("iteration", 1)
            return "Draft written" if iteration == 1 else f"Draft {iteration} written"

        if name == "reflect":
            critique = result.get("critique")
            if critique is None:
                return None
            total, unsupported = len(critique.claims), len(critique.unsupported_claims)
            if critique.passed:
                return f"✅ Fact-check passed: {total - unsupported} of {total} claims supported"
            issues = []
            if unsupported:
                issues.append(_plural(unsupported, "unsupported claim"))
            if critique.missing_aspects:
                issues.append(_plural(len(critique.missing_aspects), "unanswered part"))
            if critique.problems:
                issues.append(_plural(len(critique.problems), "other issue"))
            iteration = _get(state, "iteration", 0) or 0
            next_step = "keeping the best draft" if iteration >= self.max_iterations else "revising"
            return f"⚠️ Fact-check: {', '.join(issues)} → {next_step}"

        if name == "search_aspects":
            n = len(result.get("documents") or []) - len(_get(state, "documents", []) or [])
            return f"Searched for the missing parts: {_plural(n, 'new source')}"

        return None
