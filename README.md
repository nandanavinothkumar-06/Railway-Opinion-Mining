# 🚆 Railway Opinion Mining Dashboard

An interactive Python application for analyzing railway passenger feedback using text mining, sentiment analysis, and keyword-based categorization.

## 📌 Project Overview

Railway Opinion Mining analyzes unstructured passenger feedback to identify commonly discussed topics, estimate sentiment, and categorize railway service-related issues. The project applies text preprocessing, Bag-of-Words (BoW), Term Frequency–Inverse Document Frequency (TF-IDF), Truncated Singular Value Decomposition (TruncatedSVD), and rule-based tagging.

An interactive Streamlit dashboard presents the analysis through charts, a WordCloud, summary metrics, searchable feedback, and an analyzer for new comments. The project demonstrates how text mining techniques can support the exploration and interpretation of passenger feedback.

## 🎯 Objectives

- Process and analyze railway-related feedback.
- Identify frequently occurring words and phrases.
- Represent text using BoW and TF-IDF.
- Visualize high-dimensional text features using TruncatedSVD.
- Estimate positive, negative, and neutral sentiment.
- Categorize comments into railway service-related topics.
- Build an interactive dashboard for feedback exploration.
- Support filtering, searching, and downloading analysis results.

## ✨ Key Features

### 1. Dataset Overview
Displays feedback records, unique cleaned comments, average words per comment, and the estimated percentage of negative feedback.

### 2. WordCloud Visualization
Highlights frequently occurring terms in the cleaned feedback corpus.

### 3. Sentiment Analysis
Displays sentiment distributions and available compound scores. VADER is used for rule-based sentiment estimation when available.

### 4. Text Feature Extraction
- **Bag-of-Words (BoW):** Represents documents using term counts.
- **TF-IDF:** Assigns weights to terms based on their frequency and distribution.
- **Top-Term Analysis:** Ranks frequent terms and terms with high average TF-IDF scores.

### 5. Dimensionality Reduction
Uses TruncatedSVD to project TF-IDF features into two dimensions for exploratory visualization.

### 6. Issue Categorization
Uses keyword-based rules to identify candidate topics, including:
- Punctuality
- Cleanliness
- Staff service
- Ticketing
- Safety
- Food
- Facilities
- Infrastructure
- Information
- Accessibility
- Appreciation
- Other or unmatched feedback

A comment can receive multiple tags when it matches multiple categories.

### 7. Feedback Explorer
Allows users to search comments, filter feedback by sentiment and issue category, inspect records, and download filtered results.

### 8. New Feedback Analyzer
Accepts a new railway-related comment and displays an estimated sentiment and candidate issue tags.

### 9. CSV Export
Supports downloading filtered feedback, BoW terms, TF-IDF terms, candidate terms, and tag-count summaries.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| scikit-learn | BoW, TF-IDF, and TruncatedSVD |
| VADER Sentiment | Rule-based sentiment estimation |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| Matplotlib | WordCloud rendering |
| WordCloud | Frequent-term visualization |
| CSV | Dataset input and analysis exports |

## 📂 Project Structure

```text
Railway-Opinion-Mining/
├── app.py
├── requirements.txt
├── README.md
└── railway_feedback_analyzed.csv   # Optional
```

The CSV file can be placed beside `app.py` for automatic loading or uploaded through the dashboard sidebar.

## 📊 Dataset Description

The dashboard was tested with an analyzed CSV dataset containing railway-related feedback.

Observed results from the development version:

| Metric | Observed Value |
|---|---:|
| Feedback records | 4,504 |
| Unique cleaned comments | 4,330 |
| Average words per comment | 32.3 |
| Estimated negative feedback | 38.5% |
| BoW feature matrix | 4,504 × 5,000 |
| TF-IDF feature matrix | 4,504 × 5,000 |
| Two-component SVD explained variance ratio | Approximately 0.008 (0.8%) |

These figures describe the previously tested dataset and dashboard configuration. Results can vary with different data or settings.

**Dataset source:** Add the original dataset name, source URL, access date, and license after verifying the provider's information. These details have not been established in the project information available here.

## ⚙️ Methodology

### Step 1: Data Loading
Load the railway feedback CSV using pandas. Alternatively, upload the dataset through the dashboard.

### Step 2: Text Preprocessing
Use the cleaned-text column when available. Otherwise, normalize the original text by converting it to lowercase, removing URLs and selected punctuation, and standardizing whitespace.

### Step 3: Bag-of-Words
Apply `CountVectorizer` to create a document-term matrix and rank terms according to their frequency.

### Step 4: TF-IDF
Apply `TfidfVectorizer` to create weighted text features and identify terms with high average TF-IDF scores.

### Step 5: Dimensionality Reduction
Apply TruncatedSVD to the TF-IDF matrix and visualize the resulting two-dimensional representation.

### Step 6: Sentiment Estimation
Use sentiment labels and scores available in the dataset. When needed and available, VADER calculates compound scores and assigns positive, negative, or neutral labels using standard thresholds.

### Step 7: Keyword-Based Tagging
Match feedback against an editable set of keywords and phrases. Comments can receive multiple tags or be assigned `other_or_unmatched` when no category matches.

### Step 8: Dashboard Visualization
Display summary metrics, sentiment charts, WordCloud, text features, tag distributions, and feedback records.

### Step 9: Export
Allow users to download selected results and filtered feedback as CSV files.

## 🚀 Installation and Execution

### Prerequisites

- Python 3.10 or another version compatible with the installed dependencies
- pip
- Git, if cloning the repository
- An analyzed railway feedback CSV

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Railway-Opinion-Mining.git
cd Railway-Opinion-Mining
```

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository URL.

### 2. Create a virtual environment

For Ubuntu or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### 5. Load the dataset

Place `railway_feedback_analyzed.csv` in the same directory as `app.py`, or upload the file through the sidebar.

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Upload `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Visit https://share.streamlit.io/.
3. Sign in using GitHub.
4. Create a new app and select the repository and branch.
5. Set the application file path to `app.py`.
6. Click **Deploy**.
7. Test the deployed dashboard and share its URL with your faculty.

If the application must load a dataset automatically, make sure the CSV is available in the deployed repository. Alternatively, users can upload an approved dataset through the dashboard.

## 📈 Results and Discussion

The project demonstrates a workflow for transforming unstructured feedback into numerical text features, sentiment estimates, and interpretable issue categories.

During development, expanding the keyword taxonomy reduced unmatched tag occurrences from 2,182 to 1,906, a reduction of approximately 12.65%. This indicates that the expanded rules matched more comments or tag occurrences, but it does not establish improved classification accuracy.

The observed negative-feedback proportion was 38.5%. This is a sentiment-analysis output and should not be interpreted as a verified measurement of overall railway service quality.

The two-component TruncatedSVD visualization captured approximately 0.8% of feature variation in the observed output. Consequently, the scatter plot is intended for exploratory analysis rather than demonstrating clear separation between sentiment classes.

## ⚠️ Limitations

- VADER sentiment estimates have not been validated against an independently human-labelled test set.
- Accuracy, precision, recall, and F1-score have not been established.
- Keyword-based tagging can miss context, synonyms, spelling variations, sarcasm, and indirect complaints.
- Frequent terms do not necessarily represent the most severe passenger concerns.
- The two-dimensional SVD visualization may preserve only a small fraction of the original feature variation.
- Dataset representativeness, source, and licensing depend on the original provider.
- Tags generated for new messages may differ from the tags already stored in the dataset.

## 🔮 Future Enhancements

- Evaluate sentiment estimates against manually labelled feedback.
- Develop supervised machine learning models for issue classification.
- Explore transformer-based models for contextual sentiment analysis.
- Add topic modelling and emerging-issue detection.
- Improve spelling normalization and multilingual text processing.
- Add temporal and route-based comparisons when suitable data is available.
- Share a common tagging function between batch processing and interactive analysis.
- Introduce evaluation metrics and validation reports.

## 🎓 Conclusion

Railway Opinion Mining demonstrates how text mining and NLP techniques can be applied to unstructured passenger feedback. By combining BoW, TF-IDF, TruncatedSVD, sentiment estimation, keyword-based categorization, and interactive visualization, the project provides a practical way to explore recurring themes in railway-related comments.

The dashboard supports exploratory analysis and academic demonstration. Further validation, verified dataset provenance, and evaluation against labelled feedback would be necessary before using the results for operational decisions.

## 👩‍💻 Author

**Name:** Nandana V.  
**Programme:** Integrated M.Tech in Computer Science and Engineering (Data Science)  
**Institution:** Vellore Institute of Technology (VIT)


*Railway Opinion Mining Dashboard — An academic project in text mining, sentiment estimation, and interactive data analysis.*
