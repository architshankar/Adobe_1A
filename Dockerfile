FROM --platform=linux/amd64 python:3.12.7-slim

# Set working directory
WORKDIR /app

# Copy files
COPY process_pdfs.py .
COPY model_randomforest.joblib .
COPY decision_tree_model.joblib .
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Set command to auto-run script when container starts
CMD ["python", "process_pdfs.py"]
