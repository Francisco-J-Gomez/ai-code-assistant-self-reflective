"""The agent graph as a Graphviz diagram, coloured by what a run did - the
chat UI's live version of the `langgraph dev` view.

Drawn from the compiled graph (`graph.get_graph(xray=True)`), so it can't
drift from the code. Pure: takes the node counters kept by
agent/progress.ProgressTracker and returns DOT source, which the UI renders
with `st.graphviz_chart` (client-side, no Graphviz install needed).

    running  -> blue, the step(s) in progress
    visited  -> green, with "x2" when a node ran more than once
    neither  -> grey, the paths this run didn't take
"""
from collections import Counter

START, END = "__start__", "__end__"

# Apple-ish palette, matching the chat UI's styling.
RUNNING = {"fillcolor": "#0071e3", "color": "#0071e3", "fontcolor": "#ffffff"}
VISITED = {"fillcolor": "#e8f6ec", "color": "#34c759", "fontcolor": "#1d1d1f"}
IDLE = {"fillcolor": "#ffffff", "color": "#d2d2d7", "fontcolor": "#86868b"}


def _name(node_id: str) -> str:
    """'corpus_search:grade_docs' -> 'grade_docs' (what the progress
    events call it)."""
    return node_id.rsplit(":", 1)[-1]


def _attrs(**attrs) -> str:
    return "[" + ", ".join(f'{k}="{v}"' for k, v in attrs.items()) + "]"


def graph_dot(drawable, visited: Counter | dict | None = None, running: Counter | dict | None = None) -> str:
    """DOT source for `drawable` (a `get_graph(xray=True)` result)."""
    visited, running = visited or {}, running or {}
    finished = any(visited.get(n) for n in ("finalize", "decline"))

    def style(node_id: str) -> dict:
        name = _name(node_id)
        if node_id == START:
            return VISITED if visited or running else IDLE
        if node_id == END:
            return VISITED if finished else IDLE
        if running.get(name):
            return RUNNING
        return VISITED if visited.get(name) else IDLE

    def label(node_id: str) -> str:
        name = _name(node_id)
        if node_id in (START, END):
            return name.strip("_")
        count = visited.get(name, 0)
        return f"{name} ×{count}" if count > 1 else name

    # The subgraph's own __start__ / __end__ add nothing visually: edges into
    # its __start__ are redirected to the node it leads to.
    sub_starts = {nid for nid in drawable.nodes if nid.endswith(":" + START)}
    sub_ends = {nid for nid in drawable.nodes if nid.endswith(":" + END)}
    entry = {e.source: e.target for e in drawable.edges if e.source in sub_starts}

    lines = [
        "digraph agent {",
        '  graph [rankdir=TB, bgcolor="transparent", nodesep=0.25, ranksep=0.35, fontname="Helvetica"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=1.4, height=0.35];',
        '  edge [color="#c7c7cc", arrowsize=0.6];',
    ]
    clusters: dict[str, list[str]] = {}
    for node_id in drawable.nodes:
        if node_id in sub_starts or node_id in sub_ends:
            continue
        shape = {"shape": "oval", "fontsize": 9, "height": 0.25} if node_id in (START, END) else {}
        lines.append(f'  "{node_id}" {_attrs(label=label(node_id), **style(node_id), **shape)};')
        if ":" in node_id:
            clusters.setdefault(node_id.split(":", 1)[0], []).append(node_id)

    for cluster, members in clusters.items():
        active = running.get(cluster)
        colour = RUNNING["color"] if active else ("#34c759" if visited.get(cluster) else "#d2d2d7")
        lines.append(f'  subgraph "cluster_{cluster}" {{')
        lines.append(f'    graph {_attrs(label=f"{cluster} (per sub-query)", style="rounded,dashed", color=colour, fontsize=10, fontcolor="#6e6e73")};')
        lines += [f'    "{m}";' for m in members]
        lines.append("  }")

    for e in drawable.edges:
        if e.source in sub_starts or e.target in sub_ends:
            continue
        target = entry.get(e.target, e.target)
        attrs = {"style": "dashed"} if e.conditional else {}
        lines.append(f'  "{e.source}" -> "{target}" {_attrs(**attrs) if attrs else ""};')
    lines.append("}")
    return "\n".join(lines)
