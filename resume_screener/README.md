# ResumeIQ — AI Resume Screener

A beginner-friendly Flask web app that screens and ranks resumes against a
job description using both **TF-IDF keyword matching** and **S-BERT similarity**
via sentence-transformers.

---

## Project Structure

```
resume_screener/
├── app.py                  ← Flask backend (all logic lives here)
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Frontend HTML
├── static/
│   ├── style.css           ← Stylesheet
│   └── main.js             ← Frontend JavaScript
└── uploads/                ← Temporary resume storage (auto-created)
```

---

## Setup & Run (Step-by-Step)

### 1. Prerequisites
- Python 3.10 or higher
- pip

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB)
> on **first run**. This requires internet access the first time only.

### 4. Run the app
```bash
python app.py
```

You should see:
```
Loading sentence-transformer model...
Model loaded successfully.
 * Running on http://127.0.0.1:5000
```

### 5. Open in browser
Visit: **http://localhost:5000**

---

## How to Use

1. **Upload Resumes** — Drag & drop or browse for PDF files (multiple supported)
2. **Paste Job Description** — Enter the role requirements or keywords
3. **Adjust Weights** — Control how much TF-IDF vs S-BERT matching contributes
4. **Click "Analyze Resumes"** — Wait ~5–15 seconds for results
5. **View Rankings** — Top 3 candidates are highlighted; download buttons available

---

## Matching Methods Explained

| Method | Library | How it works |
|--------|---------|-------------|
| **TF-IDF Keyword** | scikit-learn | Converts text to weighted term vectors; cosine similarity measures keyword overlap |
| **S-BERTing** | sentence-transformers | Encodes meaning into 384-dim vectors; captures synonyms and context |
| **Combined Score** | — | Weighted average (default 40% TF-IDF + 60% S-BERT) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| PDF shows 0% match | Resume may be image-based (scanned) — use text-based PDFs |
| Slow first run | Model downloading; subsequent runs are fast |
| Port already in use | Change `port=5000` in `app.py` to another port |

---

## Bonus Features Included

- ✅ Matched keyword highlighting per resume
- ✅ Download shortlisted resumes directly from the UI
- ✅ Adjustable TF-IDF / S-BERT weight sliders
- ✅ Drag & drop file upload with file management
- ✅ Error handling for invalid PDFs and empty inputs
