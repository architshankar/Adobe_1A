
import json
import os
from collections import Counter
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import fitz  # PyMuPDF
import joblib
import pandas as pd
from ftfy import fix_text
from PyPDF2 import PdfReader
from langdetect import detect

# ------------------------ CONFIGURATION CONSTANTS ------------------------- #

PDF_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

MODEL_PATH = Path("/app/model_randomforest.joblib")
level_model_path = Path("/app/decision_tree_model.joblib")


HEADING_LABEL = 1          # value assigned to "heading" by the RF model
FAST_MODE = False          # True ⇒ limit to first 20 pages
MAX_WORKERS = min(mp.cpu_count(), 4)  # Number of parallel workers

# -------------------------- ORIGINAL HELPERS ----------------------------- #
_nlp = None
_tfidf_vectorizer = None
_model = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def get_tfidf_vectorizer():
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _tfidf_vectorizer = TfidfVectorizer()
    return _tfidf_vectorizer

def get_model():
    """Cache and return the loaded heading detection model"""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

# Add second model for heading level classification
_level_model = None
def get_level_model():
    """Cache and return the loaded heading level classification model"""
    global _level_model
    if _level_model is None:
        # level_model_path = BASE_DIR / "decision_tree_model.joblib"
        level_model_path = Path("/app/decision_tree_model.joblib")
        _level_model = joblib.load(level_model_path)
    return _level_model

import re
NUMBERING_PATTERNS = [
    re.compile(r'^\d+[\.\)]'),
    re.compile(r'^[A-Z]+\.'),
    re.compile(r'^\([a-zA-Z0-9]+\)'),
    re.compile(r'^\d+(\.\d+)+')
]
HEADING_KEYWORDS = frozenset({
    "chapter", "section", "article", "part", "appendix", "figure", "table",
    "introduction", "abstract", "methodology", "methods", "results", "findings",
    "discussion", "conclusion", "conclusions", "references", "bibliography",
    "overview", "summary", "analysis", "implementation", "executive summary",
    "contents", "index", "preface", "one", "two", "three", "four", "five",
    "about", "background", "approach", "data", "model", "framework"
})
_STOPWORDS = None
def get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = get_nlp().Defaults.stop_words
    return _STOPWORDS

def is_numbering(text):
    return int(any(p.match(text.strip()) for p in NUMBERING_PATTERNS))

def contains_colon(text):
    return int(':' in text)

def is_uppercase(text):
    return int(text.strip().isupper())

def contains_stopwords_ratio(text):
    words = text.lower().split()
    if not words: return 0
    stopwords = get_stopwords()
    return sum(w in stopwords for w in words) / len(words)

def starts_with_heading_keyword(text):
    words = text.strip().lower().split()
    return int(words and words[0].rstrip(".:") in HEADING_KEYWORDS)

def clean_text_line(line):
    line = fix_text(line)
    # Only remove control characters, keep Unicode text
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', line).strip()

def is_non_english(text):
    try:
        lang = detect(text)
        return lang != "en"
    except Exception:
        return True  # If detection fails, assume non-English

def extract_features_from_pdf(pdf_path, max_pages=None):
    if not pdf_path.exists():
        return pd.DataFrame()
    records = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return pd.DataFrame()

    pages_to_process = min(len(doc), max_pages) if max_pages else len(doc)

    for page_num in range(pages_to_process):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        spans = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    spans.extend(line["spans"])

        if not spans:
            continue

        avg_font = sum(s["size"] for s in spans) / len(spans)
        size_counts = Counter(round(s["size"], 1) for s in spans)
        color_counts = Counter(s["color"] for s in spans)
        page_width = page.rect.width
        prev_y1 = None
        lines = []

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                if not line["spans"]:
                    continue

                text = " ".join(fix_text(s["text"]) for s in line["spans"]).strip()
                if not text or len(text) < 2:
                    continue

                sizes = [s["size"] for s in line["spans"]]
                fonts = [s["font"] for s in line["spans"]]
                colors = [s["color"] for s in line["spans"]]
                x0, y0, x1, y1 = line["bbox"]
                spacing = y0 - prev_y1 if prev_y1 is not None else 0
                prev_y1 = y1

                lines.append({
                    "text": clean_text_line(text),
                    "sizes": sizes,
                    "fonts": fonts,
                    "colors": colors,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "spacing": spacing
                })

        for i, line in enumerate(lines):
            text = line["text"]
            words = text.split()
            if len(words) > 15:
                continue
            max_f = max(line["sizes"])
            font_ratio = max_f / avg_font
            if font_ratio < 0.8:
                continue
            max_f = max(line["sizes"])
            rfont = round(max_f, 1)
            is_centered = int(
                abs(line["x0"] - (page_width - (line["x1"] - line["x0"])) / 2) < 10
            )
            has_following_paragraph = 0
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                line_spacing = next_line["y0"] - line["y1"]
                indent_diff = next_line["x0"] - line["x0"]
                next_text = next_line["text"]
                if (line_spacing > 2 and indent_diff >= 10 and
                    len(next_text.split()) >= 5 and
                    next_text and not next_text[0].isupper()):
                    has_following_paragraph = 1
            font_name = line["fonts"][0] if line["fonts"] else ""
            font_color = line["colors"][0] if line["colors"] else 0

            is_non_eng = is_non_english(text)

            records.append({
                "pdf_name": Path(pdf_path).name,
                "page": page_num + 1,
                "text": text,
                "num_words": len(text.split()),
                "num_chars": len(text),
                "has_numbering": is_numbering(text),
                "contains_colon": contains_colon(text),
                "is_uppercase": is_uppercase(text),
                "contains_stopwords_ratio":  0.0 if is_non_eng else contains_stopwords_ratio(text),
                "starts_with_keyword":  0 if is_non_eng else starts_with_heading_keyword(text),
                "indentation": round(line["x0"], 2),
                "font_ratio": round(max_f / avg_font, 3),
                "font_size_freq_ratio": round(size_counts[rfont] / len(spans), 3),
                "font_name": font_name,
                "font_name_freq": round(line["fonts"].count(font_name) / len(line["fonts"]), 3) if line["fonts"] else 0.0,
                "is_bold": int("bold" in font_name.lower()),
                "font_color": font_color,
                "font_color_is_unique": int(color_counts[font_color] < 0.05 * len(spans)),
                "block_width": round(line["x1"] - line["x0"], 2),
                "block_height": round(line["y1"] - line["y0"], 2),
                "line_spacing_above": round(line["spacing"], 2),
                "x0": round(line["x0"], 2),
                "y0": line["y0"],
                "is_centered": is_centered,
                "has_following_paragraph": has_following_paragraph
            })

    doc.close()

    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame()

    if not df.empty:
        try:
            tfidf = get_tfidf_vectorizer()
            tfidf_matrix = tfidf.fit_transform(df["text"])
            df["tfidf_score"] = tfidf_matrix.max(axis=1).toarray().ravel().round(4)
        except Exception:
            df["tfidf_score"] = 0.0

    return df

def assign_level(row):
    fr = row["font_ratio"]
    if fr > 1.5:  return "H1"
    if fr > 1.3:  return "H2"
    if fr > 0.8:  return "H3"
    return "H4"

def infer_on_pdf(
    pdf_path: Path,
    model_path: Path,
    out_json: Path,
    heading_label: int,
    fast_mode: bool = False
) -> None:
    # Extract title (same as before)
    try:
        reader = PdfReader(pdf_path)
        meta_title = reader.metadata.title or "" if reader.metadata else ""
    except Exception:
        meta_title = ""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return
    page0 = doc[0]
    half_y = page0.rect.height * 0.5
    candidates = [
        ((blk["bbox"][2]-blk["bbox"][0])*(blk["bbox"][3]-blk["bbox"][1]),
         " ".join(fix_text(span["text"])
                  for line in blk.get("lines", [])
                  for span in line["spans"]).strip())
        for blk in page0.get_text("dict")["blocks"]
        if blk.get("lines") and blk["bbox"][3] <= half_y
    ]
    doc.close()
    title = meta_title if any(t==meta_title for _,t in candidates) else \
            max(candidates, default=(0,""))[1] if candidates else ""

    # Load both models (cached)
    try:
        clf_heading = get_model()
        clf_level = get_level_model()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required model file missing: {e}")

    # Extract features
    df = extract_features_from_pdf(pdf_path, max_pages=20 if fast_mode else None)
    if df.empty:
        return

    # Ensure all expected features are present for heading detection
    for feat in clf_heading.feature_names_in_:
        if feat not in df.columns:
            df[feat] = 0.0

    # Stage 1: Heading detection
    df["is_heading_pred"] = clf_heading.predict(df[clf_heading.feature_names_in_])

    # Select only heading rows
    heading_df = df[df["is_heading_pred"] == heading_label].copy()

    if heading_df.empty:
        # No headings found, save empty outline
        outline = []
    else:
        # Ensure all expected features are present for heading level model
        for feat in clf_level.feature_names_in_:
            if feat not in heading_df.columns:
                heading_df[feat] = 0.0

        # Stage 2: Heading level classification (H1, H2, H3, H4)
        heading_df["heading_level"] = clf_level.predict(heading_df[clf_level.feature_names_in_])

        # Prepare outline with ML-predicted levels
        heads = (
            heading_df.drop_duplicates("text")
                      .sort_values(["page", "y0"])
        )
        outline = [
            {
                "level": f"H{int(r['heading_level'])}",  # Use ML-predicted level
                "text": r["text"],
                "page": int(r["page"])
            }
            for _, r in heads.iterrows() if r["text"] != title
        ]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump({"title": title, "outline": outline}, fh,
                  ensure_ascii=False, indent=2)

def process_single_pdf(pdf_file):
    """Process a single PDF file - wrapper for multithreading"""
    try:
        out_path = OUTPUT_DIR / f"{pdf_file.stem}.json"
        infer_on_pdf(pdf_file, MODEL_PATH, out_path, HEADING_LABEL, FAST_MODE)
        return True
    except Exception as e:
        print(f"Error processing {pdf_file.name}: {e}")
        return False

def process_all_pdfs() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Required model not found at {MODEL_PATH}")
    
    # Check for heading level model
    # level_model_path = BASE_DIR / "decision_tree_model.joblib"
    # if not level_model_path.exists():
    #     raise FileNotFoundError(f"Required heading level model not found at {level_model_path}")
    
    # Check for heading level model
    if not level_model_path.exists():
        raise FileNotFoundError(f"Required heading level model not found at {level_model_path}")

    
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory {PDF_DIR} is missing.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        return
    
    # print(f"Processing {len(pdf_files)} PDF files using {MAX_WORKERS} workers...")
    
    # Pre-load shared resources in main thread to avoid repeated loading
    get_model()          # Load heading detection model
    get_level_model()    # Load heading level classification model
    get_nlp()
    get_tfidf_vectorizer()
    get_stopwords()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_pdf, pdf_files))
    
    successful = sum(results)
    # print(f"Successfully processed {successful}/{len(pdf_files)} PDF files")

if __name__ == "__main__":
    process_all_pdfs()