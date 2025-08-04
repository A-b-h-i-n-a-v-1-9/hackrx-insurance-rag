import os
import json
import tempfile
import requests
import fitz  # PyMuPDF
import faiss
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

app = FastAPI()

# === Config ===
EMBED_MODEL = "all-MiniLM-L6-v2"
GGUF_MODEL_PATH = "models/phi-2.Q4_K_M.gguf"
API_KEY = "cache-cache"

# === Load models ===
model = SentenceTransformer(EMBED_MODEL)
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=1024,
    n_threads=6,
    temperature=0.2,
    top_p=0.95
)

# === Text chunking ===
def chunk_text(text, chunk_size=700, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks or ["No meaningful content found."]

# === LLM-based answer generation ===
def generate_answer(context: str, question: str) -> str:
    try:
        prompt = f"""You are a helpful assistant. Answer the question below using only the information in the policy. 
Respond in 1-2 clear, concise sentences. Avoid any unnecessary information.

Policy Info:
\"\"\"{context}\"\"\"

Q: {question}
A:"""
        response = llm(prompt, max_tokens=180, stop=["\n", "Q:", "A:"], echo=False)
        return response["choices"][0]["text"].strip() or "Sorry, no answer found."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# === Main HackRx Endpoint ===
@app.post("/hackrx/run")
async def run_rag(request: Request):
    try:
        # 🔐 Authorization
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer ") or auth.split()[1] != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # 📥 Parse payload
        payload = await request.json()
        pdf_url = payload.get("documents")
        questions = payload.get("questions", [])

        if not pdf_url or not questions:
            raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' field")

        # 📄 Download PDF
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF")

        # ✂️ Extract text
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        text = "\n".join(page.get_text().replace('\xa0', ' ') for page in doc)
        chunks = chunk_text(text)

        # 🔎 FAISS Search
        embeddings = model.encode(chunks).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # 🤖 Answer each question
        answers = []
        for question in questions:
            try:
                q_embed = model.encode([question]).astype('float32')
                _, I = index.search(q_embed, 3)
                context = "\n\n".join(chunks[i] for i in I[0])
                answer = generate_answer(context, question)
                answers.append(answer)
            except Exception as e:
                answers.append(f"Error answering '{question}': {str(e)}")

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
