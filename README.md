# langchain_rag

A compact **Retrieval-Augmented Generation (RAG)** document Q&A pipeline built
with [LangChain](https://python.langchain.com/). Point it at a folder of your own
`.txt` / `.md` files and ask questions — answers are grounded in the documents,
not the model's memory.

## Why RAG?

A language model only knows what it saw during training. Ask it about your
internal docs, a new library, or last week's meeting notes and it will either
say it doesn't know or, worse, make something up. RAG fixes this by fetching the
most relevant pieces of *your* data at question time and handing them to the
model as context. The model then answers from that context instead of from
memory — which keeps answers current, private, and far less prone to
hallucination.

## How it works

The pipeline runs in two phases.

**Indexing (once, at startup):**

1. **Load** — every `.txt` / `.md` file under `docs/` is read via `DirectoryLoader`.
2. **Chunk** — `RecursiveCharacterTextSplitter` breaks each document into ~800-character overlapping pieces so retrieval is granular and no chunk exceeds the model's context budget.
3. **Embed & store** — each chunk is turned into a vector with `OpenAIEmbeddings` and saved in a **FAISS** index for fast similarity search.

**Answering (per question):**

4. **Retrieve** — the question is embedded and FAISS returns the top 4 most similar chunks.
5. **Prompt** — those chunks are formatted (with their source filenames) into the context slot of a `ChatPromptTemplate`.
6. **Generate** — a chat model answers, constrained by a system prompt to use *only* the retrieved context and to admit when the answer isn't there.

```
docs/ ──load──> chunks ──embed──> FAISS index
                                      │
question ──embed──> similarity search ┘──> top-k chunks
                                              │
                    prompt(context, question) ┘──> chat model ──> answer
```

## What it showcases

- **Document loading & chunking** — `DirectoryLoader` + `RecursiveCharacterTextSplitter`
- **Embeddings & vector search** — `OpenAIEmbeddings` stored in a **FAISS** index, used as a retriever
- **LCEL composition** — the whole pipeline is wired declaratively with the pipe operator: `retriever → prompt → chat model → output parser`
- **Grounded answers** — a system prompt constrains the model to the retrieved context and cites source filenames

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env         # then add your OPENAI_API_KEY
python rag.py                # interactive Q&A over ./docs
python rag.py "What is RAG?" # one-shot question
```

Drop your own `.txt` or `.md` files into `docs/` and re-run — the index is
rebuilt from whatever is there.

## Project layout

| File | Purpose |
|------|---------|
| `rag.py` | The full pipeline: load → chunk → embed → retrieve → answer |
| `docs/` | Your knowledge base (ships with a sample file so it runs out of the box) |
| `requirements.txt` | Pinned LangChain, FAISS, and OpenAI dependencies |
| `.env.example` | Template for your `OPENAI_API_KEY` |

## Stack

`Python` · `LangChain (LCEL)` · `FAISS` · `OpenAI`
