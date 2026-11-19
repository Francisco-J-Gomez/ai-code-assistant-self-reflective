"""Turn the graph's final state into what the UI shows: a deduplicated
sources list and a corpus / web / not-grounded badge.

Reads `documents` and `route` from the state. (Before Phase 1 these were
parsed out of search_papers / tavily_search ToolMessages; retrieval is a
graph node now, so those messages no longer exist.)
"""
from agent.state import Document


def sources_from_documents(documents: list[Document]) -> list[dict]:
    """Citations for this turn, de-duplicated by URL (several chunks of the
    same paper can be retrieved)."""
    sources = []
    seen_urls = set()
    for doc in documents:
        if doc.url in seen_urls:
            continue
        seen_urls.add(doc.url)
        source = {"kind": doc.kind, "title": doc.title, "url": doc.url}
        if doc.kind in ("corpus", "arxiv"):
            source["arxiv_id"] = doc.arxiv_id
        sources.append(source)
    return sources


def answer_badge(route: str | None) -> dict:
    """Which path the graph actually took this turn (not what the answer's
    prose claims): "web" means the answer includes a live search - the
    direct web route, or fallback_research (web and/or live arXiv) after the
    corpus wasn't enough."""
    if route == "web":
        return {"icon": "🌐", "label": "Includes live search", "kind": "web"}
    if route == "corpus":
        return {"icon": "📄", "label": "Grounded in the paper corpus", "kind": "corpus"}
    return {"icon": "❌", "label": "Not grounded — no relevant source found", "kind": "none"}


def self_check_note(critique, drafts: int) -> str | None:
    """One line on the reflection loop's verdict for the answer shown
    (Phase 4). None when there was no draft (declined questions)."""
    if critique is None or drafts == 0:
        return None
    revised = f" after {drafts - 1} revision{'s' if drafts > 2 else ''}" if drafts > 1 else ""
    if critique.passed:
        return f"✅ Fact-checked against its sources{revised}"
    return f"⚠️ Didn't fully pass the fact-check{revised} — see the note at the end"
