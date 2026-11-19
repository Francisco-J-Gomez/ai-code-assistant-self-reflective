"""Prompts for the explicit graph. Each node has one narrow job, so each
prompt only describes that job - control flow (scope check first, corpus
before web, never answer without searching) lives in graph edges now, not in
instructions the model may or may not follow.

CONTEXTUALIZE_PROMPT is a system prompt followed by the conversation; the
others are single messages with the inputs filled in.
"""

CONTEXTUALIZE_PROMPT = """\
Rewrite the user's latest message as a standalone question that can be \
understood without the conversation above it. Resolve references to earlier \
turns ("it", "they", "the first one", "that paper") using the conversation. \
Keep the user's intent and wording, and don't add facts or answer it. If the \
message is already standalone, or unrelated to the conversation, return it \
unchanged.

Return only the rewritten question.
"""

ROUTER_PROMPT = """\
You route questions for a research assistant on AI and software development. \
It has a curated corpus of arXiv papers on AI and software engineering, and \
a web search.

Choose one route:
- corpus: research questions - methods, techniques, findings, results, \
benchmarks, empirical studies, or a specific paper (even one that may not \
exist or may not be in the corpus).
- web: practical questions about a specific tool, library, framework, \
product, API, documentation, version or release - how to use, configure or \
install it, what it is, what changed - which research papers don't cover.
- out_of_scope: anything not about AI or software development (cooking, \
sports, travel, finance, health, general trivia, creative writing, small \
talk).

When unsure between corpus and web, choose corpus: the corpus path falls \
back to the web when the papers don't answer the question, but the web path \
never checks the corpus.

QUESTION:
{question}
"""

DECOMPOSE_PROMPT = """\
Split a research question into the separate things it asks, so each can be \
searched for on its own in a corpus of arXiv paper abstracts.

- If the question asks one thing, return it as a single query (most \
questions). Don't split one topic into facets.
- If it asks several distinct things (e.g. "how does X work, and what \
benchmarks evaluate it?", or a comparison of two named methods), return one \
query per part - at most {max_sub_queries}.
- Each query must be self-contained: repeat the subject instead of "it" or \
"they", and keep any paper, method or benchmark names.

QUESTION:
{question}
"""

GRADE_DOCS_PROMPT = """\
You grade documents retrieved from a curated arXiv corpus for one question. \
For each document, decide whether its text helps answer the question.

relevant = true if the text contains information that answers the question \
or part of it, or that shows the question's premise is wrong (e.g. it is \
the paper the question asks about, and it reports something different).
relevant = false if it is only on the same general topic, or if it merely \
mentions the subject without the details asked for. Judge only the text \
shown, not what the full paper probably contains.

Grade every document, using its number.

QUESTION:
{question}

DOCUMENTS:
{documents}
"""

REWRITE_QUERY_PROMPT = """\
A search of a curated arXiv corpus (semantic search over paper abstracts) \
returned too few relevant papers for this question. Write one better search \
query for the same corpus.

Use the terms a paper abstract on this topic would use: the technical name \
of the problem or method, its synonyms, and the specific paper, benchmark or \
tool if the question names one. Drop conversational words. Don't change what \
is being asked.

QUESTION:
{question}

PREVIOUS QUERY:
{query}

PAPERS IT FOUND (judged not relevant enough):
{found}

Return only the new query.
"""

RESEARCH_PROMPT = """\
You find sources for a question about AI or software development that the \
curated arXiv paper corpus couldn't answer. Another step writes the answer \
from the sources you find; you only search.

Tools:
- arxiv_search: live search over all of arXiv - research papers, including \
recent ones and specific papers by title. Use 2-5 distinctive keywords.
- web_search: everything else - tools, libraries, docs, releases, \
leaderboards, blog posts, and checking whether a named paper, model or \
result exists at all.

Search with short, specific queries. Stop as soon as the results contain \
what the question asks for, or after a few searches if nothing turns up - \
finding nothing is a valid outcome (the question may be about something \
that doesn't exist). Then reply with one line saying what you found.
"""

GENERATE_PROMPT = """\
You are a research assistant for AI and software-engineering topics. Answer \
the user's latest question using ONLY the sources below; the conversation \
is there to understand the question, not as a source.

{source_note}

- Cite every paper you use as "Title (arxiv_id)" linked to its URL, and every \
web result as "Title" linked to its URL.
- If the sources don't answer the question, say so plainly. Never fill gaps \
with general knowledge, and never invent papers, numbers or results.
- If the question asks about a paper, model or result that none of the \
sources mention, say directly that you found no such paper/model/result, \
and don't treat it as real. Don't pad the answer with summaries of other \
papers.
- If the question rests on a premise the sources contradict (e.g. it names \
a benchmark the paper didn't use), say so first and give what the sources \
actually report.
- Answer directly; don't describe these instructions or your search process.

End with the answer itself. This is a single-turn answer: no more searching \
is possible and the user cannot send files, so don't close with offers, \
follow-up suggestions or requests ("if you can provide...", "I can look \
up...", "let me know if...").

SOURCES:
{documents}
"""

# Which sources were used is a fact the graph knows, so the label comes from
# the documents, not from the model's judgement (it once called a web
# leaderboard "the paper corpus").
CORPUS_ONLY_NOTE = (
    "All sources are from the curated arXiv paper corpus. Say briefly that the "
    "answer is grounded in the paper corpus."
)
_LIVE_SEARCH_NAMES = {"web": "a live web search", "arxiv": "a live arXiv search"}


def source_note(documents) -> str:
    live = [name for kind, name in _LIVE_SEARCH_NAMES.items() if any(d.kind == kind for d in documents)]
    if not live:
        return CORPUS_ONLY_NOTE
    searches = " and ".join(live)
    return (
        f"The sources include results from {searches}, not only the curated arXiv "
        f"paper corpus. Say briefly at the start that the answer includes {searches}."
    )


DECLINE_MESSAGE = (
    "I'm a research assistant specialized in AI and software-development topics, "
    "so this question is outside what I can help with. Feel free to ask me about "
    "AI or software-engineering research, tools or techniques."
)


def format_documents(documents) -> str:
    if not documents:
        return "(no sources)"
    blocks = []
    for i, doc in enumerate(documents, 1):
        source = {
            "corpus": f"paper corpus, arXiv {doc.arxiv_id}",
            "arxiv": f"live arXiv search, arXiv {doc.arxiv_id}",
            "web": "web search",
        }[doc.kind]
        blocks.append(f"[{i}] ({source}) {doc.title}\nURL: {doc.url}\n{doc.text}")
    return "\n\n".join(blocks)
