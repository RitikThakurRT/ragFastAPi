from fastapi import FastAPI, HTTPException
import ollama
import chromadb
import time
import concurrent.futures

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
        # 🔍 RETRIEVE
        results = collection.query(
            query_texts=[question],
            n_results=2,  # keep small for low RAM
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

        models = ["phi3", "qwen2.5:0.5b"]  # keep max 2 for your system

        responses = []

        # ⚡ Parallel execution (LIMITED threads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(query_model, model, augmented_prompt)
                for model in models
            ]

            for future in concurrent.futures.as_completed(futures):
                responses.append(future.result())

        return {
            "question": question,
            "responses": responses,
            "context_used": documents,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_pdf_parallel:app", host="0.0.0.0", port=8000, reload=True)
