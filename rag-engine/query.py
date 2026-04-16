# rag-engine/query.py
import os
import json
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

PORT = int(os.getenv('PORT', 5001))
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")

# ==========================================
# 🧠 LAZY LOADING VARIABLES
# ==========================================
embedder = None
index = None
chunks = None

def initialize_ai():
    """Loads the heavy AI models ONLY when the first question is asked."""
    global embedder, index, chunks
    if embedder is None:
        print("⏳ First question received! Waking up AI Models...")
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_grad_enabled(False)
        
        if os.path.exists(INDEX_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            print("✅ FAISS Database loaded.")
        else:
            print(f"❌ Error: Missing {INDEX_FILE}")

        embedder = SentenceTransformer(MODEL_NAME, device='cpu')
        print("✅ AI Embedding Model loaded on CPU.")

SYSTEM_PROMPT = (
    "You are an elite, highly accurate AI Assistant for University PG College Secunderabad (UPGCS). "
    "You must follow these absolute rules:\n"
    "1. CONVERSATIONAL ETIQUETTE: If the user sends a simple greeting, respond warmly. Do not use the fallback phrase for greetings.\n"
    "2. STRICT FACTUALITY: Answer ONLY using the facts provided in the Context Information.\n"
    "3. NAME COLLISION RULE: If multiple people share a name (like Sandhya), treat them as separate individuals.\n"
    "4. ENTITY ISOLATION: Never blend names, roles, or titles.\n"
    "5. UNKNOWN INFO: If the answer is not in the context, reply exactly with: 'I'm sorry, I don't have information on that.' Do not guess."
)

def generate_groq_response(question, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant", 
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context Information:\n{context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.0 
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return True, response.json()['choices'][0]['message']['content']
        return False, None
    except Exception:
        return False, None

def generate_gemini_response(question, context):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    combined_prompt = f"{SYSTEM_PROMPT}\n\nContext Information:\n{context}\n\nQuestion: {question}"
    payload = {
        "contents": [{"parts": [{"text": combined_prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return True, response.json()['candidates'][0]['content']['parts'][0]['text']
        return False, "The UPGCS Assistant is experiencing high traffic. Please try again. 🎓"
    except Exception:
        return False, "The UPGCS Assistant is experiencing high traffic. Please try again. 🎓"

@app.route('/query', methods=['POST'])
def query_rag():
    # THE MAGIC TRICK: Load AI only when a student actually asks a question!
    initialize_ai()
    
    data = request.json
    question = data.get('question')

    if not question: return jsonify({"error": "No question provided"}), 400

    question_embedding = embedder.encode([question])
    k = 12
    distances, indices = index.search(np.array(question_embedding).astype('float32'), k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
    
    success, answer = generate_groq_response(question, context)
    if success:
        print("⚡ Answered using GROQ API.")
    else:
        print("🔄 Routing to Engine 2 (Gemini)...")
        success, answer = generate_gemini_response(question, context)

    return jsonify({
        "answer": answer,
        "sources": relevant_chunks,
        "confidence": float(1 / (1 + distances[0][0]))
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "database": "waiting for first query"})

if __name__ == '__main__':
    print(f"🚀 Starting fast on port {PORT}. AI is sleeping until needed.")
    app.run(host='0.0.0.0', port=PORT, debug=False)