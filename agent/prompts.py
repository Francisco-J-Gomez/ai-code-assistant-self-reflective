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

CHECK_CORPUS_PROMPT = """\
You decide whether papers retrieved from a curated arXiv corpus are enough \
to answer a user's question, or whether a web search is needed.

answers_question = true only if the text below contains information that \
actually answers the question (not merely the same general topic). Judge \
only the text shown, not what the full paper probably contains: a paper \
that says it covers something without giving the details doesn't answer a \
question about those details. If the question asks how to use a specific \
tool, library, API or product, or about a specific paper or number, and the \
text doesn't give that answer, answer false.

QUESTION:
{question}

RETRIEVED PAPERS:
{documents}
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
WITH_WEB_NOTE = (
    "The sources include results from a live web search, not only the curated "
    "arXiv paper corpus. Say briefly at the start that the answer includes a "
    "live web search."
)


def source_note(documents) -> str:
    return WITH_WEB_NOTE if any(d.kind == "web" for d in documents) else CORPUS_ONLY_NOTE

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
        source = f"paper corpus, arXiv {doc.arxiv_id}" if doc.kind == "corpus" else "web search"
        blocks.append(f"[{i}] ({source}) {doc.title}\nURL: {doc.url}\n{doc.text}")
    return "\n\n".join(blocks)
