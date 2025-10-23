import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np

app = Flask(__name__)
CORS(app)

# === Config ===
PDF_PATH = os.getenv("PDF_PATH", "ISO_9001_2015.pdf")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

# === PDF Loader ===
def read_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size].strip())
    return chunks

print("🔹 Loading PDF...")
text = read_pdf_text(PDF_PATH)
chunks = chunk_text(text)
print(f"✅ Loaded {len(chunks)} chunks from {PDF_PATH}")

# === Build TF-IDF Index ===
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
tfidf = vectorizer.fit_transform(chunks)
print("✅ TF-IDF index ready.")

# === Retrieval ===
def retrieve_context(question, top_k=5):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(tfidf, q_vec).ravel()
    idx = np.argsort(-sims)[:top_k]
    return "\n\n".join(chunks[i] for i in idx)

# === Endpoint ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    q = data.get("question", "").strip()
    if not q:
        return jsonify({"error": "No question provided"}), 400

    context = retrieve_context(q)

    system_prompt = (
    "You are a helpful assistant specializing in ISO 9001:2015. "
    "Use the document text provided to answer clearly, accurately, and conversationally. "
    "Focus on explaining concepts in plain language, not just quoting text. "
    "If the answer isn't in the document, use your understanding of ISO principles to give helpful context — "
    "but indicate when something is an interpretation. "
    "When relevant, mention the clause number or section that supports your answer."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {q}\n\nDocument:\n{context}"}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "message": "POST /ask {question:'...'}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
