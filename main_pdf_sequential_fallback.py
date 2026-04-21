from fastapi import FastAPI, HTTPException
import ollama
import chromadb
import time

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()

# 🔹 Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="pdf_collection",
    embedding_function=ef,
)


# 🔹 Function to query a model
def query_model(model_name, prompt):
    start = time.time()

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    end = time.time()

    return {
        "model": model_name,
        "answer": response["message"]["content"],
        "latency_sec": round(end - start, 2),
    }


@app.get("/ask")
def ask(question: str):
    try:
        # 🔍 RETRIEVE (keep small for low RAM)
        results = collection.query(
            query_texts=[question],
            n_results=2,
        )

        documents = results.get("documents", [[]])[0]
        context = "\n\n".join(documents) if documents else "No relevant context found."

        # 🧠 AUGMENT
        augmented_prompt = f"""
Answer ONLY using the provided context.
If the answer is not present, say:
"I don't have enough information from the provided context."

Context:
{context}

Question: {question}
"""

        responses = []

        # 🥇 Step 1: Primary model (fast + efficient)
        primary_model = "phi3"
        res1 = query_model(primary_model, augmented_prompt)
        responses.append(res1)

        # 🧠 Step 2: Fallback logic (simple heuristic)
        # If answer is too short or uncertain → call backup model
        if (
            len(res1["answer"].strip()) < 50
            or "don't have enough information" in res1["answer"].lower()
        ):
            backup_model = "qwen2.5:0.5b"
            res2 = query_model(backup_model, augmented_prompt)
            responses.append(res2)

        return {
            "question": question,
            "responses": responses,   # includes model attribution
            "context_used": documents,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
