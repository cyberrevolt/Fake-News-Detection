# Fake News Detection

A Flask web application that classifies news text with trained machine learning models and offers optional Gemini-assisted article analysis. It also includes user accounts, prediction history, model development notebooks, datasets, and a project report.

## Features

- Predict whether submitted news text is likely fake or real using the saved TF-IDF and classification model.
- Optionally analyze an article with Gemini, including a summary, key claims, and verification advice.
- Register and sign in to save and review prediction history.
- Explore data preparation, feature extraction, training, evaluation, and final model testing in Jupyter notebooks.

Model predictions are statistical classifications and should not be treated as independent verification of an article's claims.

## Requirements

- Python 3.10 or newer
- A PostgreSQL database (or another SQLAlchemy-compatible database URL)
- Optional Gemini API key for AI-assisted analysis
- NLTK English stopwords and WordNet data

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following values:

```dotenv
SECRET_KEY=replace-with-a-long-random-secret
DATABASE_URL=postgresql://username:password@localhost:5432/fake_news_detection
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.5-flash
```

`GEMINI_API_KEY` is optional. Without it, the classifier remains available and Gemini analysis reports that it is not configured. Keep `.env` private; it is excluded from Git.

Download the required NLTK resources once in a Python session:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

## Run the application

```bash
python app.py
```

Open the local address printed by Flask in your browser. Ensure the database configured in `DATABASE_URL` is running and accessible before starting the app.

## Project layout

```text
app.py                     Flask application and prediction routes
src/preprocessing.py       Text preprocessing utilities
templates/                 Jinja templates for the web interface
static/                    Stylesheets and static assets
model/                     Serialized models, vectorizers, and evaluation results
dataset/                   Source and processed datasets
notebooks/                 Data exploration, training, and evaluation notebooks
report_figures/            Figures used in the project report
Fake_News_Detection_Project_Report.docx
requirements.txt
```

## Notes

- The project includes large datasets and serialized model files. GitHub rejects individual files larger than 100 MB in ordinary Git pushes; use Git LFS or external dataset storage for those files.
- The Gemini integration provides supporting analysis, not fact-checking or a guarantee of accuracy.
