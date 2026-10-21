# Railway Opinion Mining Dashboard

## What this dashboard displays
- Dataset summary and feedback filters
- Sentiment distribution
- Issue-category distribution when the dataset has a category column
- Bag-of-Words frequency chart
- TF-IDF term ranking
- TruncatedSVD two-dimensional visualization
- Refined tag counts and candidate tag tables
- Searchable feedback table and CSV downloads
- Interactive analysis of a new feedback message

## Prepare the input file
1. Run all cells in `Railway_Opinion_Mining_Fixed.ipynb`.
2. Download `railway_feedback_analyzed.csv` from the notebook.
3. Keep that CSV beside `app.py`, or upload it in the dashboard sidebar.
4. Optionally keep `overall_candidate_tags.csv` beside the app or upload it in the sidebar.

The app expects a text column such as `text_original`, `Tweet`, `text`, `feedback`, or `comment`. It will use the `sentiment`, `sentiment_score`, and `refined_tags_text` columns if present.

## Run locally (Windows)
Open Command Prompt or PowerShell in this folder and run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

If `py` is unavailable, use `python` instead.

## Run in Google Colab (optional)
Streamlit is designed to run as a web app. The simplest option is to run it locally on your computer. If your course requires Colab, you will need a tunnel or deployment service to expose the app, which is not included here.

## Notes for your report
- VADER sentiment is lexicon-based and should be described as an estimate.
- Candidate tags are rule-based and need manual review.
- TruncatedSVD coordinates are exploratory; the axes do not represent specific semantic categories.
- Do not claim model accuracy unless it is calculated from held-out labeled test data.
