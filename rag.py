"""
LangChain RAG — a compact document Q&A pipeline.

Showcases the core LangChain building blocks:
  * Document loading + chunking (RecursiveCharacterTextSplitter)
  * Embeddings + a FAISS vector store as a retriever
  * An LCEL chain wiring retriever -> prompt -> chat model -> parser
  * Answers grounded in retrieved context, with source citations

Usage:
    python rag.py                       # interactive Q&A over ./docs
    python rag.py "your question here"  # one-shot question
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent / "docs"
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """You are a precise assistant. Answer the question using ONLY the \
context below. If the answer is not in the context, say you don't know — do not \
invent facts.

Context:
{context}"""


def load_documents() -> list[Document]:
    """Load every .txt / .md file under ./docs."""
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.[tm][xd]*",  # .txt and .md
        loader_cls=TextLoader,
        show_progress=False,
    )
    docs = loader.load()
    if not docs:
        sys.exit(f"No documents found in {DOCS_DIR}. Add .txt or .md files first.")
    return docs


def build_retriever():
    """Chunk the docs, embed them, and expose a similarity-search retriever."""
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))
    return vector_store.as_retriever(search_kwargs={"k": 4})


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks (with their source) for the prompt context."""
    return "\n\n".join(
        f"[{Path(d.metadata.get('source', '?')).name}]\n{d.page_content}" for d in docs
    )


def build_chain(retriever):
    """Compose the RAG pipeline with LCEL: retrieve -> prompt -> model -> parse."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    model = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


def main() -> None:
    load_dotenv()
    print("Indexing documents in ./docs ...")
    retriever = build_retriever()
    chain = build_chain(retriever)

    if len(sys.argv) > 1:  # one-shot mode
        question = " ".join(sys.argv[1:])
        print(f"\nQ: {question}\nA: {chain.invoke(question)}")
        return

    print("Ready. Ask a question (Ctrl-C or empty line to quit).\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        print(f"A: {chain.invoke(question)}\n")


if __name__ == "__main__":
    main()
