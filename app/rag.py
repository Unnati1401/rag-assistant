import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")


def get_embeddings(kind):
    """model -> (embeddings, collection_name)."""
    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small"), "docs_openai"
    elif kind == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return (
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            "docs_minilm",
        )
    else:
        raise ValueError(f"Unknown embedding kind: {kind!r} (use 'openai' or 'hf')")


def build_retriever(kind=DEFAULT_EMBED_MODEL, k=4, persist_directory="./chroma_db"):
    """Retriever pointed at the collection matching the embedding model."""
    embeddings, collection_name = get_embeddings(kind)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def make_llm(model="gpt-4o-mini", temperature=0):
    return ChatOpenAI(model=model, temperature=temperature)


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using ONLY the "
        "context provided. If the answer is not in the context, say you "
        "don't know. Be concise.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def format_context(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def answer(question, retriever, llm):
    """Retrieve, generate, and return (answer_text, retrieved_docs)."""
    docs = retriever.invoke(question)
    context = format_context(docs)
    messages = _PROMPT.invoke({"context": context, "question": question})
    text = llm.invoke(messages).content
    return text, docs