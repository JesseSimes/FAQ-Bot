# FAQ-Bot
# RAG Pipeline — Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that answers questions using a local folder of markdown documents as the knowledge base.

## Architecture

| Stage | Tool | Purpose |
|---|---|---|
| **LLM (main processing power)** | Gemini (`langchain_google_genai`) | Generates the final natural-language answer |
| **Chunking** | `RecursiveCharacterTextSplitter` | Splits documents into chunks (`chunk_size=1000`, `chunk_overlap=200` — 20% overlap for better context continuity) |
| **Embedding** | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) | Converts text chunks into vectors representing their meaning |
| **Storage** | ChromaDB (`langchain_chroma`) | Stores and searches the embedded vectors |
| **Retrieval** | LangChain Classic (`RetrievalQA`) | Combines retrieval (search) with generation (Gemini) into one pipeline |

## Pipeline Flow

```
.md files
   ↓ (DirectoryLoader / TextLoader)
Raw documents
   ↓ (RecursiveCharacterTextSplitter, 1000 chars, 200 overlap)
Chunks
   ↓ (HuggingFaceEmbeddings — all-MiniLM-L6-v2)
Vectors
   ↓ (Chroma.from_documents)
Vector Store (persisted to disk)
   ↓ (vectorstore.as_retriever(k=3))
Retriever
   ↓ (RetrievalQA.from_chain_type)
Question → Top 3 relevant chunks → Gemini → Answer
```

## Why These Choices

- **20% chunk overlap** — prevents ideas/sentences from being cut off at chunk boundaries, ensuring retrieved chunks retain full context.
- **all-MiniLM-L6-v2** — fast, free, runs locally on CPU, no API cost, well-suited for general-purpose semantic search (384-dimension vectors).
- **ChromaDB** — lightweight local vector database, persists to disk, integrates natively with LangChain.
- **RetrievalQA (langchain_classic)** — combines retrieval + generation in one call, returns both the answer and its source documents for verification.

## Example Usage

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

response = qa_chain.invoke({"query": "What is the refund policy?"})

print(response["result"])              # Gemini's answer
print(response["source_documents"])    # Chunks used to generate it
```
