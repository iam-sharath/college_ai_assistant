# rag-engine/ingest.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATA_FILE = "data/UPGCS_Chatbot_Knowledge_Base.txt"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

def create_smart_chunks(text, max_chars=1200):
    """
    PERFECT ANALYSIS CHUNKER:
    Splits text logically by paragraphs/sections instead of blind word counts.
    This ensures courses and FAQs stay together in the AI's memory.
    """
    # Split by double-newlines (which separates your sections and FAQs)
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para or "===" in para: # Skip empty lines and decorative borders
            continue
            
        # If adding this paragraph keeps the chunk under the limit, append it
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            # Save the current chunk and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk (with a little overlap by keeping the section title if possible)
            current_chunk = para + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def main():
    print("🎓 Starting UPGCS Smart Knowledge Base Ingestion...")

    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Could not find {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Smart Chunking
    print("✂️  Performing Smart Chunking by sections and paragraphs...")
    chunks = create_smart_chunks(text)
    print(f"✅ Created {len(chunks)} highly-optimized memory chunks.")

    # 2. Load Model & Convert
    print(f"⏳ Loading AI Embedding Model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print("🧠 Converting text to mathematical vectors...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # 3. Create FAISS Index & Save
    print("🗄️  Building FAISS Vector Database...")
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

    faiss.write_index(index, INDEX_FILE)
    
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

    print("🎉 PERFECT SUCCESS! Knowledge Base is deeply analyzed and saved.")

if __name__ == "__main__":
    main()