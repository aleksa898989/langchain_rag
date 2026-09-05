# Sample knowledge base

This file is here so the demo works out of the box. Replace it with your own
`.txt` or `.md` files and re-run — the pipeline will index whatever it finds.

## What is RAG?

Retrieval-Augmented Generation (RAG) grounds a language model's answers in your
own documents. Instead of relying only on what the model memorized during
training, the system retrieves the most relevant chunks of your data at query
time and passes them to the model as context. This reduces hallucination and
lets the model answer questions about private or up-to-date information.

## How this project works

1. Documents in the `docs/` folder are loaded and split into overlapping chunks.
2. Each chunk is embedded into a vector and stored in a FAISS index.
3. At query time, the question is embedded and the four most similar chunks are
   retrieved.
4. Those chunks become the context in a prompt, and a chat model produces an
   answer grounded in that context.

## LCEL

LangChain Expression Language (LCEL) is the declarative way to compose chains
using the pipe (`|`) operator. In this project the chain is:
retriever -> prompt -> chat model -> output parser.
