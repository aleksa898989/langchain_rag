# langchain_rag

A compact **Retrieval-Augmented Generation (RAG)** document Q&A pipeline built
with [LangChain](https://python.langchain.com/). Point it at a folder of your own
`.txt` / `.md` files and ask questions — answers are grounded in the documents,
not the model's memory.

## What it showcases

- **Document loading & chunking** — `DirectoryLoader` + `RecursiveCharacterTextSplitter`
- **Embeddings & vector search** — `OpenAIEmbeddings` stored in a **FAISS** index, used as a retriever
- **LCEL composition** — the pipeline is wired declaratively: `retriever → prompt → chat model → output parser`
- **Grounded answers** — a system prompt constrains the model to the retrieved context and admits when it doesn't know

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your OPENAI_API_KEY
python rag.py               # interactive Q&A over ./docs
python rag.py "What is RAG?" # one-shot question
```

Drop your own `.txt` or `.md` files into `docs/` and re-run — the index is
rebuilt from whatever is there.

## How it works

```
docs/ ──load──> chunks ──embed──> FAISS index
                                      │
question ──embed──> similarity search ┘──> top-k chunks
                                              │
                    prompt(context, question) ┘──> chat model ──> answer
```

## Stack

`Python` · `LangChain (LCEL)` · `FAISS` · `OpenAI`
