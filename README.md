# 🧠 PDF Heading Extraction using Machine Learning

This project processes PDF documents to detect **headings** and predict their **hierarchical level (H1-H4)** using machine learning. The tool supports batch processing, and outputs structured outlines in JSON format for each PDF.

---

## 📌 Approach

We use a **two-stage ML pipeline** to extract structured outlines from unstructured PDF text:

1. **Heading Detection:**  
   A **RandomForest** classifier predicts whether a line of text is a heading or body content based on font features, layout cues, and text semantics.

2. **Heading Level Classification:**  
   A second model (**DecisionTree**) predicts heading levels (H1, H2, H3, H4) for lines identified as headings. These levels help build a meaningful document structure.

### Key Features Extracted

- Font size, ratio, and boldness  
- Text alignment and indentation  
- Presence of stopwords and heading keywords  
- Capitalization and punctuation  
- Color uniqueness, spacing, and TF-IDF scores  
- Language detection to skip non-English content  

---

## 📦 Models & Libraries

### ML Models Used

| Stage                  | Model Type        | File                                 |
|------------------------|-------------------|--------------------------------------|
| Heading Detection      | RandomForest       | `model_randomforest.joblib`          |
| Heading Level Detection| DecisionTree       | `decision_tree_model.joblib`         |

### Python Libraries

- `PyMuPDF (fitz)` – for PDF parsing  
- `PyPDF2` – for metadata extraction  
- `scikit-learn` – for ML model inference  
- `spaCy` – for stopword and token analysis  
- `ftfy` – for Unicode text fixing  
- `langdetect` – for detecting non-English content  
- `pandas`, `joblib`, `concurrent.futures`, `multiprocessing`  

---

## 🚀 Expected Execution

### Python Execution

Run locally using Python 3.12:

```bash
python process_pdfs.py
```

All PDFs in `/app/input` will be processed and their outlines saved in `/app/output`.

---

### 🐳 Docker Execution

You can run the project inside a Docker container using:

**1. Build the image:**

```bash
docker build --platform linux/amd64 -t mysolutionname:mytag .
```

**2. Run the container:**

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```


---

## 🛠️ How It Works

1. The script loads and caches ML models and NLP resources.  
2. Each PDF is read using `fitz` and processed page-by-page.  
3. Features are extracted per line.  
4. Lines are classified as **headings or body text**.  
5. Headings are further classified into **H1–H4** levels.  
6. A structured outline is saved as a `.json` file with:

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Problem Statement", "page": 2}
  ]
}
```

---

## ✅ Example Output

For a file `sample.pdf`, the result will be:

```
/app/output/sample.json
```

Containing the title and detected outline in structured JSON format.

---

## 📎 Notes

- Processing uses **parallel threads** to handle multiple PDFs efficiently.  
