"""
AI Resume Screener - Flask Backend
===================================
Handles PDF upload, text extraction, TF-IDF keyword matching,
and S-BERT similarity using sentence-transformers.
"""

import os
import re
import json
import numpy as np
import pdfplumber
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─── App Configuration ────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max per file
ALLOWED_EXTENSIONS = {"pdf"}

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the sentence-transformer model once at startup (avoids reloading per request)
print("Loading sentence-transformer model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

# ─── Utility Functions ────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a .pdf extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract clean text from a PDF file using pdfplumber.
    Handles multi-page PDFs and returns joined text.
    """
    text_pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text.strip())
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
        return ""

    full_text = "\n".join(text_pages)
    # Basic cleanup: collapse multiple whitespace/newlines
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]+", " ", full_text)
    return full_text.strip()


def tfidf_similarity(resume_texts: list[str], job_desc: str) -> list[float]:
    """
    Compute TF-IDF cosine similarity between each resume and the job description.
    Returns a list of scores (0.0 – 1.0) in the same order as resume_texts.
    """
    if not resume_texts or not job_desc.strip():
        return [0.0] * len(resume_texts)

    corpus = resume_texts + [job_desc]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [0.0] * len(resume_texts)

    job_vec = tfidf_matrix[-1]           # last entry = job description
    resume_vecs = tfidf_matrix[:-1]      # all others = resumes
    scores = cosine_similarity(resume_vecs, job_vec).flatten()
    return scores.tolist()


def sbert_similarity(resume_texts: list[str], job_desc: str) -> list[float]:
    """
    Compute S-BERT cosine similarity using sentence-transformers (all-MiniLM-L6-v2) embeddings.
    Returns a list of scores (0.0 – 1.0).
    """
    if not resume_texts or not job_desc.strip():
        return [0.0] * len(resume_texts)

    all_texts = resume_texts + [job_desc]
    embeddings = sbert_model.encode(all_texts, convert_to_numpy=True)
    job_emb = embeddings[-1].reshape(1, -1)
    resume_embs = embeddings[:-1]
    scores = cosine_similarity(resume_embs, job_emb).flatten()
    # Clamp to [0, 1] since cosine can technically go slightly negative
    scores = np.clip(scores, 0.0, 1.0)
    return scores.tolist()


def extract_keywords(text: str, job_desc: str, top_n: int = 10) -> list[str]:
    """
    Extract keywords from the job description that also appear in the resume text.
    Returns a list of matched keywords (up to top_n).
    """
    # Tokenise job description: lowercase words with 3+ chars
    job_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", job_desc.lower()))
    resume_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

    # Common English stopwords to ignore
    stopwords = {
        "the", "and", "for", "are", "was", "with", "this", "that", "have",
        "from", "will", "our", "you", "your", "not", "all", "can", "has",
        "its", "more", "been", "they", "their", "but", "also", "which",
        "when", "use", "using", "used", "able", "make", "made", "such",
        "than", "into", "her", "his", "him", "she", "who", "what", "how",
    }
    job_words -= stopwords
    matched = sorted(job_words & resume_words)[:top_n]
    return matched


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main endpoint: receives PDF files + job description,
    runs TF-IDF and S-BERT matching, returns ranked results.
    """
    # 1. Validate inputs
    if "resumes" not in request.files:
        return jsonify({"error": "No resume files uploaded."}), 400

    job_desc = request.form.get("job_description", "").strip()
    if not job_desc:
        return jsonify({"error": "Job description cannot be empty."}), 400

    tfidf_weight = float(request.form.get("tfidf_weight", 0.4))
    sbert_weight = float(request.form.get("sbert_weight", 0.6))

    files = request.files.getlist("resumes")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Please upload at least one PDF resume."}), 400

    # 2. Save uploaded PDFs and extract text
    saved_files = []
    resume_texts = []
    filenames = []

    for file in files:
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(filepath)
            text = extract_text_from_pdf(filepath)

            if not text:
                # Skip unreadable PDFs but report them
                saved_files.append({"filename": fname, "error": "Could not extract text."})
                continue

            saved_files.append(fname)
            resume_texts.append(text)
            filenames.append(fname)
        else:
            return jsonify({"error": f"Invalid file type: {file.filename}. Only PDFs allowed."}), 400

    if not resume_texts:
        return jsonify({"error": "No readable PDFs found. Ensure files are text-based PDFs."}), 400

    # 3. Compute similarity scores
    tfidf_scores = tfidf_similarity(resume_texts, job_desc)
    sbert_scores   = sbert_similarity(resume_texts, job_desc)

    # 4. Combine scores (weighted average)
    combined = [
        tfidf_weight * t + sbert_weight * s
        for t, s in zip(tfidf_scores, sbert_scores)
    ]

    # 5. Build result list and sort by combined score descending
    results = []
    for i, fname in enumerate(filenames):
        matched_kw = extract_keywords(resume_texts[i], job_desc)
        results.append({
            "filename":       fname,
            "tfidf_score":    round(tfidf_scores[i] * 100, 2),
            "sbert_score": round(sbert_scores[i] * 100, 2),
            "combined_score": round(combined[i] * 100, 2),
            "matched_keywords": matched_kw,
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)

    # Assign ranks
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return jsonify({"results": results, "total": len(results)})


@app.route("/download/<filename>")
def download_resume(filename: str):
    """Serve a previously uploaded resume for download."""
    safe_name = secure_filename(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name, as_attachment=True)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
