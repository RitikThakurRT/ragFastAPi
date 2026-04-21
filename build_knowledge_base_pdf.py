import re
import pdfplumber
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# -----------------------------
# Step 1: Extract text from PDF
# -----------------------------
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# -----------------------------
# Step 2: Clean text
# -----------------------------
def clean_text(text):
    # Normalize newlines
    text = re.sub(r"\n+", "\n", text)

    # Merge broken lines into sentences
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text.strip()


# -----------------------------
# Step 3: Word-based chunking
# -----------------------------
def chunk_by_words(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


# -----------------------------
# Step 4: Load + Process PDF
# -----------------------------
pdf_path = "aiResearchPaper.pdf"

text = extract_pdf_text(pdf_path)
text = clean_text(text)
print(text)

chunks = chunk_by_words(text, chunk_size=150, overlap=30)

print(f"Loaded {len(chunks)} chunks from PDF")


# -----------------------------
# Step 5: Initialize ChromaDB
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="pdf_collection",
    embedding_function=ef,
)


# -----------------------------
# Step 6: Store embeddings
# -----------------------------
collection.add(
    ids=[f"chunk{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=[{"source": "pdf", "chunk_index": i} for i in range(len(chunks))],
)

print("✅ Done: PDF processed, chunked (word-based), and stored in ChromaDB!")
