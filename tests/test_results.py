from agent.results import answer_badge, sources_from_documents
from agent.state import Document

PAPER = Document(kind="corpus", title="A Paper", url="http://arxiv.org/abs/2301.12345", text="...", arxiv_id="2301.12345")
WEB = Document(kind="web", title="A Blog", url="https://example.com/post", text="...")


def test_sources_from_corpus_document():
    assert sources_from_documents([PAPER]) == [
        {"kind": "corpus", "title": "A Paper", "url": "http://arxiv.org/abs/2301.12345", "arxiv_id": "2301.12345"}
    ]


def test_sources_from_web_document_has_no_arxiv_id():
    assert sources_from_documents([WEB]) == [{"kind": "web", "title": "A Blog", "url": "https://example.com/post"}]


def test_sources_from_live_arxiv_document_keep_arxiv_id():
    live = Document(kind="arxiv", title="New", url="https://arxiv.org/abs/2611.00001", text="...", arxiv_id="2611.00001")
    assert sources_from_documents([live]) == [
        {"kind": "arxiv", "title": "New", "url": "https://arxiv.org/abs/2611.00001", "arxiv_id": "2611.00001"}
    ]


def test_sources_dedupe_by_url_keeping_order():
    second_chunk = PAPER.model_copy(update={"text": "another chunk"})
    assert [s["url"] for s in sources_from_documents([PAPER, WEB, second_chunk])] == [PAPER.url, WEB.url]


def test_sources_empty():
    assert sources_from_documents([]) == []


def test_badge_per_route():
    assert answer_badge("web")["kind"] == "web"
    assert answer_badge("corpus")["kind"] == "corpus"
    assert answer_badge("decline")["kind"] == "none"
    assert answer_badge(None)["kind"] == "none"


def test_self_check_note():
    from agent.results import self_check_note
    from agent.schemas import ClaimCheck, Critique

    ok = Critique(claims=[ClaimCheck(claim="a", supported=True)], missing_aspects=[], problems=[])
    bad = Critique(claims=[ClaimCheck(claim="a", supported=False)], missing_aspects=[], problems=[])
    assert self_check_note(ok, 1) == "✅ Fact-checked against its sources"
    assert self_check_note(ok, 2) == "✅ Fact-checked against its sources after 1 revision"
    assert self_check_note(bad, 3).startswith("⚠️ Didn't fully pass the fact-check after 2 revisions")
    assert self_check_note(None, 0) is None  # declined: no draft
