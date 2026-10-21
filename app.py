
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Optional sentiment and WordCloud dependencies are handled gracefully.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Railway Opinion Mining",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(120, 130, 145, 0.07);
        border: 1px solid rgba(120, 130, 145, 0.18);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# TEXT AND COLUMN UTILITIES
# ---------------------------------------------------------
def clean_text(value):
    """Normalize text for analysis without changing the original comment."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_column(data, candidates):
    """Find a column using case-insensitive candidate names."""
    lookup = {str(c).strip().lower(): c for c in data.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def safe_text_series(data, column):
    if column is None:
        return pd.Series("", index=data.index, dtype=str)
    return data[column].fillna("").astype(str)


TEXT_CANDIDATES = [
    "text_original", "tweet", "text", "feedback", "comment",
    "review", "message", "opinion", "content"
]

CLEAN_CANDIDATES = [
    "clean_text", "cleaned_text", "processed_text", "cleaned_comment"
]

SENTIMENT_CANDIDATES = [
    "sentiment", "sentiment_label", "predicted_sentiment", "polarity"
]

SCORE_CANDIDATES = [
    "sentiment_score", "compound", "compound_score", "vader_score"
]

CATEGORY_CANDIDATES = [
    "type", "issue_type", "category", "feedback_type", "issue_category"
]

TAG_CANDIDATES = [
    "refined_tags_text", "tags", "predicted_tags", "tag"
]


# ---------------------------------------------------------
# SENTIMENT AND TAGGING FOR NEW COMMENTS
# ---------------------------------------------------------
TAG_GROUPS = {
    "punctuality": [
        "late", "delay", "delayed", "waiting", "waited",
        "on time", "running late", "slow train", "speed",
        "slow", "schedule", "departure", "arrival"
    ],
    "cleanliness": [
        "clean", "dirty", "garbage", "toilet", "hygiene",
        "dustbin", "litter", "smell", "filthy", "unhygienic"
    ],
    "staff_service": [
        "staff", "attendant", "rude", "behaviour", "behavior",
        "helpful", "help", "conductor", "employee", "support"
    ],
    "ticketing": [
        "ticket", "booking", "refund", "reservation",
        "payment", "irctc", "cancellation", "cancelled"
    ],
    "safety": [
        "safety", "unsafe", "security", "harassment",
        "theft", "robbery", "emergency", "danger"
    ],
    "food": [
        "food", "meal", "catering", "water", "pantry",
        "tea", "coffee", "pricey food", "stale"
    ],
    "facilities": [
        "seat", "coach", "ac", "air conditioning",
        "fan", "light", "berth", "charging", "washroom",
        "bedding", "blanket"
    ],
    "infrastructure": [
        "station", "platform", "track", "bridge",
        "infrastructure", "railway line", "foot overbridge"
    ],
    "information": [
        "announcement", "information", "display board",
        "timetable", "update", "communication", "signage"
    ],
    "accessibility": [
        "wheelchair", "accessible", "disability",
        "elderly", "ramp", "lift", "senior citizen"
    ],
    "appreciation": [
        "excellent", "wonderful", "great service",
        "thank you", "appreciate", "comfortable journey",
        "well done", "amazing"
    ],
}


def assign_tags(text):
    """Transparent keyword-based tags for a newly entered comment."""
    normalized = clean_text(text)
    found = []

    for tag, keywords in TAG_GROUPS.items():
        for keyword in keywords:
            pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
            if re.search(pattern, normalized):
                found.append(tag)
                break

    return found or ["other_or_unmatched"]


def analyze_new_feedback(text):
    """Estimate sentiment and assign candidate tags."""
    tags = assign_tags(text)

    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)["compound"]
        sentiment = (
            "Positive" if score >= 0.05
            else "Negative" if score <= -0.05
            else "Neutral"
        )
        method = "VADER compound score"
    else:
        score = None
        sentiment = "Not calculated"
        method = "Install vaderSentiment to enable sentiment scoring"

    return sentiment, score, tags, method


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
st.sidebar.title("🚆 Railway Opinion Mining")
st.sidebar.caption("Explore passenger feedback, sentiment and recurring themes.")

uploaded_file = st.sidebar.file_uploader(
    "Upload analyzed feedback CSV",
    type=["csv"],
    help="Upload railway_feedback_analyzed.csv exported from your notebook.",
)

default_file = Path(__file__).resolve().parent / "railway_feedback_analyzed.csv"

@st.cache_data(show_spinner=False)
def read_csv_file(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), low_memory=False)


df = None

try:
    if uploaded_file is not None:
        df = read_csv_file(uploaded_file.getvalue())
    elif default_file.exists():
        df = pd.read_csv(default_file, low_memory=False)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df is None:
    st.title("🚆 Railway Opinion Mining Dashboard")
    st.info(
        "Upload your analyzed CSV using the sidebar, or place "
        "'railway_feedback_analyzed.csv' in the same folder as app.py."
    )
    st.code(
        "railway_feedback_analyzed.csv\n"
        "app.py\n"
        "requirements.txt"
    )
    st.stop()

if df.empty:
    st.error("The uploaded CSV has no records.")
    st.stop()

original_col = find_column(df, TEXT_CANDIDATES)
clean_col = find_column(df, CLEAN_CANDIDATES)
sentiment_col = find_column(df, SENTIMENT_CANDIDATES)
score_col = find_column(df, SCORE_CANDIDATES)
category_col = find_column(df, CATEGORY_CANDIDATES)
tag_col = find_column(df, TAG_CANDIDATES)

if original_col is None and clean_col is None:
    st.error(
        "No recognizable feedback text column was found. "
        "Expected a column such as text_original, Tweet, text, feedback or comment."
    )
    st.write("Available columns:", list(df.columns))
    st.stop()

df = df.copy()

if original_col is None:
    original_col = clean_col

df["_original"] = safe_text_series(df, original_col)

if clean_col is not None:
    df["_clean"] = safe_text_series(df, clean_col)
else:
    df["_clean"] = df["_original"].apply(clean_text)

df["_clean"] = df["_clean"].fillna("").astype(str)
df["_original"] = df["_original"].fillna("").astype(str)

df = df.reset_index(drop=True)

# Compute sentiment only when it is missing from the source file.
if sentiment_col is not None:
    df["_sentiment"] = safe_text_series(df, sentiment_col).str.strip().str.title()
else:
    df["_sentiment"] = ""

if score_col is not None:
    df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
else:
    df["_score"] = np.nan

if VADER_AVAILABLE:
    missing_sentiment = df["_sentiment"].eq("") | df["_sentiment"].eq("Nan")
    need_score = df["_score"].isna()

    if missing_sentiment.any() or need_score.any():
        analyzer = SentimentIntensityAnalyzer()
        computed_scores = df.loc[need_score, "_original"].apply(
            lambda value: analyzer.polarity_scores(value)["compound"]
        )
        df.loc[need_score, "_score"] = computed_scores

        def score_to_sentiment(score):
            if pd.isna(score):
                return "Unknown"
            if score >= 0.05:
                return "Positive"
            if score <= -0.05:
                return "Negative"
            return "Neutral"

        df.loc[missing_sentiment, "_sentiment"] = (
            df.loc[missing_sentiment, "_score"].apply(score_to_sentiment)
        )

df["_sentiment"] = df["_sentiment"].replace("", "Unknown").fillna("Unknown")

if category_col is not None:
    df["_category"] = safe_text_series(df, category_col).replace("", "Unspecified")
else:
    df["_category"] = "Unspecified"

if tag_col is not None:
    df["_tags"] = safe_text_series(df, tag_col)
else:
    df["_tags"] = df["_original"].apply(lambda x: ", ".join(assign_tags(x)))


# ---------------------------------------------------------
# HEADER AND SIDEBAR INFORMATION
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🚆 Railway Opinion Mining Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Text analytics, sentiment estimation, '
    'keyword-based categorization and feedback exploration</div>',
    unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.write(f"**Records loaded:** {len(df):,}")
st.sidebar.write(f"**Detected text column:** `{original_col}`")
st.sidebar.write(
    "**Sentiment source:** " +
    ("CSV / VADER fallback" if sentiment_col else
     ("VADER" if VADER_AVAILABLE else "Unavailable"))
)

if not VADER_AVAILABLE:
    st.sidebar.warning(
        "VADER is not installed. Existing sentiment labels are shown when available."
    )


# ---------------------------------------------------------
# PREPARE SUMMARY DATA
# ---------------------------------------------------------
nonempty_clean = df["_clean"].str.strip().ne("")
unique_comments = df.loc[nonempty_clean, "_clean"].nunique()
word_counts = df.loc[nonempty_clean, "_clean"].apply(
    lambda value: len(value.split())
)
avg_words = float(word_counts.mean()) if len(word_counts) else 0.0

sentiment_counts = (
    df["_sentiment"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Count")
)

negative_mask = df["_sentiment"].str.lower().eq("negative")
negative_pct = float(negative_mask.mean() * 100) if len(df) else 0.0

all_text = " ".join(df.loc[nonempty_clean, "_clean"].tolist())


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
overview_tab, sentiment_tab, features_tab, tags_tab, explore_tab = st.tabs(
    [
        "📊 Overview",
        "💬 Sentiment",
        "🔤 Text Features",
        "🏷️ Tags",
        "🔎 Explore Feedback",
    ]
)


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
with overview_tab:
    st.subheader("Dataset Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Feedback records", f"{len(df):,}")
    m2.metric("Unique cleaned comments", f"{unique_comments:,}")
    m3.metric("Negative feedback", f"{negative_pct:.1f}%")
    m4.metric("Average words / comment", f"{avg_words:.1f}")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Sentiment distribution")
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            text="Count",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Comments")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Feedback by issue category")
        category_counts = (
            df["_category"].value_counts().head(15)
            .rename_axis("Category").reset_index(name="Count")
        )
        fig = px.bar(
            category_counts,
            x="Count",
            y="Category",
            orientation="h",
            text="Count",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### WordCloud: Common Terms in Railway Feedback")

    if WORDCLOUD_AVAILABLE and all_text.strip():
        wc = WordCloud(
            width=1200,
            height=500,
            background_color="white",
            max_words=150,
            collocations=False,
            min_font_size=8,
        ).generate(all_text)

        fig_wc, ax = plt.subplots(figsize=(14, 5.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc, use_container_width=True)
        plt.close(fig_wc)
    elif not WORDCLOUD_AVAILABLE:
        st.info(
            "Install WordCloud to enable this chart: "
            "`python3 -m pip install wordcloud matplotlib`"
        )
    else:
        st.info("No cleaned feedback text is available for the WordCloud.")

    st.caption(
        "Sentiment labels are estimates or labels supplied in the CSV. "
        "The WordCloud shows frequent terms, not the severity or importance of an issue."
    )


# ---------------------------------------------------------
# SENTIMENT
# ---------------------------------------------------------
with sentiment_tab:
    st.subheader("Sentiment Analysis")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Sentiment counts")
        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Sentiment score distribution")
        score_data = df["_score"].dropna()
        if not score_data.empty:
            fig = px.histogram(
                x=score_data,
                nbins=30,
                labels={"x": "Compound sentiment score", "count": "Comments"},
            )
            fig.update_layout(yaxis_title="Comments")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No numeric sentiment scores were found. "
                "If VADER is installed, missing scores are computed when possible."
            )

    st.markdown("#### Sentiment summary")
    st.dataframe(sentiment_counts, use_container_width=True, hide_index=True)

    if not df["_score"].dropna().empty:
        st.write(
            f"Mean available sentiment score: "
            f"**{df['_score'].mean():.3f}**"
        )

    st.caption(
        "VADER compound scores range from -1 (more negative) to +1 (more positive). "
        "These estimates should not be described as validated accuracy."
    )


# ---------------------------------------------------------
# TEXT FEATURES: BOW, TF-IDF, SVD
# ---------------------------------------------------------
with features_tab:
    st.subheader("Text Features: Bag-of-Words and TF-IDF")

    texts = df["_clean"].fillna("").astype(str)
    texts = texts[texts.str.strip().ne("")]

    max_features = st.slider(
        "Maximum vocabulary size",
        min_value=500,
        max_value=5000,
        value=3000,
        step=500,
    )
    top_n = st.slider(
        "Number of terms to display",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    if len(texts) < 2:
        st.warning("At least two non-empty comments are required for text features.")
    else:
        try:
            bow_vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            bow_matrix = bow_vectorizer.fit_transform(texts)
            bow_terms = bow_vectorizer.get_feature_names_out()
            bow_counts = np.asarray(bow_matrix.sum(axis=0)).ravel()

            bow_df = (
                pd.DataFrame({"Term": bow_terms, "Frequency": bow_counts})
                .sort_values("Frequency", ascending=False)
                .head(top_n)
            )

            tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            tfidf_terms = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

            tfidf_df = (
                pd.DataFrame({"Term": tfidf_terms, "Mean TF-IDF": tfidf_scores})
                .sort_values("Mean TF-IDF", ascending=False)
                .head(top_n)
            )

            a, b = st.columns(2)
            with a:
                st.markdown("#### Most frequent terms (BoW)")
                fig = px.bar(
                    bow_df.sort_values("Frequency"),
                    x="Frequency",
                    y="Term",
                    orientation="h",
                    text="Frequency",
                )
                st.plotly_chart(fig, use_container_width=True)

            with b:
                st.markdown("#### Highest mean TF-IDF terms")
                fig = px.bar(
                    tfidf_df.sort_values("Mean TF-IDF"),
                    x="Mean TF-IDF",
                    y="Term",
                    orientation="h",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.write(
                f"**BoW matrix:** {bow_matrix.shape[0]:,} documents × "
                f"{bow_matrix.shape[1]:,} features"
            )
            st.write(
                f"**TF-IDF matrix:** {tfidf_matrix.shape[0]:,} documents × "
                f"{tfidf_matrix.shape[1]:,} features"
            )

            st.download_button(
                "Download BoW top terms",
                bow_df.to_csv(index=False).encode("utf-8"),
                file_name="bow_top_terms.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download TF-IDF top terms",
                tfidf_df.to_csv(index=False).encode("utf-8"),
                file_name="tfidf_top_terms.csv",
                mime="text/csv",
            )

            st.divider()
            st.markdown("#### TruncatedSVD: Two-dimensional text projection")

            if tfidf_matrix.shape[1] >= 2 and tfidf_matrix.shape[0] >= 3:
                with st.spinner("Calculating the two-dimensional projection..."):
                    svd = TruncatedSVD(n_components=2, random_state=42)
                    coordinates = svd.fit_transform(tfidf_matrix)

                plot_df = pd.DataFrame({
                    "Component 1": coordinates[:, 0],
                    "Component 2": coordinates[:, 1],
                    "Sentiment": df.loc[texts.index, "_sentiment"].values,
                })

                fig = px.scatter(
                    plot_df,
                    x="Component 1",
                    y="Component 2",
                    color="Sentiment",
                    opacity=0.65,
                    title="Feedback projected into two dimensions",
                )
                st.plotly_chart(fig, use_container_width=True)

                explained = float(svd.explained_variance_ratio_.sum())
                st.write(
                    f"Sum of the two components' explained variance ratio: "
                    f"**{explained:.4f} ({explained * 100:.2f}%)**"
                )
                st.caption(
                    "This is an exploratory projection. A low explained-variance "
                    "ratio means the plot captures only a small portion of the "
                    "variation in the original text features."
                )
            else:
                st.info("Not enough documents or features for TruncatedSVD.")

        except ValueError as exc:
            st.warning(f"Could not calculate text features for this data: {exc}")


# ---------------------------------------------------------
# TAGS
# ---------------------------------------------------------
with tags_tab:
    st.subheader("Issue Tags and Candidate Themes")

    tag_rows = []
    for value in df["_tags"].fillna("").astype(str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        tag_rows.extend(parts)

    if tag_rows:
        tag_counts = (
            pd.Series(tag_rows, name="Tag")
            .value_counts()
            .rename_axis("Tag")
            .reset_index(name="Occurrences")
        )

        fig = px.bar(
            tag_counts.head(20).sort_values("Occurrences"),
            x="Occurrences",
            y="Tag",
            orientation="h",
            text="Occurrences",
            title="Most frequent tags",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tag_counts, use_container_width=True, hide_index=True)

        st.download_button(
            "Download tag counts",
            tag_counts.to_csv(index=False).encode("utf-8"),
            file_name="railway_tag_counts.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No tag column was found. New-feedback analysis still supports "
            "keyword-based candidate tags."
        )

    st.markdown("#### Candidate themes from TF-IDF")
    st.caption(
        "These are terms with high average TF-IDF weights, not verified issue labels."
    )

    candidate_texts = df["_clean"].fillna("").astype(str)
    candidate_texts = candidate_texts[candidate_texts.str.strip().ne("")]

    if len(candidate_texts) >= 2:
        try:
            candidate_vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=1,
                stop_words="english",
                max_df=1.0,
            )
            candidate_matrix = candidate_vectorizer.fit_transform(candidate_texts)
            candidate_terms = candidate_vectorizer.get_feature_names_out()
            candidate_scores = np.asarray(candidate_matrix.mean(axis=0)).ravel()

            candidate_df = (
                pd.DataFrame({
                    "Candidate term": candidate_terms,
                    "Mean TF-IDF": candidate_scores,
                })
                .sort_values("Mean TF-IDF", ascending=False)
                .head(25)
            )
            st.dataframe(candidate_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download candidate terms",
                candidate_df.to_csv(index=False).encode("utf-8"),
                file_name="overall_candidate_tags.csv",
                mime="text/csv",
            )
        except ValueError as exc:
            st.info(f"Candidate terms could not be generated: {exc}")
    else:
        st.info("Not enough non-empty comments to generate candidate terms.")


# ---------------------------------------------------------
# EXPLORE FEEDBACK
# ---------------------------------------------------------
with explore_tab:
    st.subheader("Search and Explore Feedback")

    search_query = st.text_input(
        "Search feedback",
        placeholder="Try: delay, cleanliness, ticket, staff...",
    )

    sentiments = sorted(df["_sentiment"].dropna().unique().tolist())
    selected_sentiments = st.multiselect(
        "Filter by sentiment",
        options=sentiments,
        default=sentiments,
    )

    categories = sorted(df["_category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Filter by issue category",
        options=categories,
        default=categories,
    )

    filtered = df.copy()

    if selected_sentiments:
        filtered = filtered[filtered["_sentiment"].isin(selected_sentiments)]
    else:
        filtered = filtered.iloc[0:0]

    if selected_categories:
        filtered = filtered[filtered["_category"].isin(selected_categories)]
    else:
        filtered = filtered.iloc[0:0]

    if search_query.strip():
        mask = (
            filtered["_original"].str.contains(
                search_query, case=False, regex=False, na=False
            )
            | filtered["_clean"].str.contains(
                search_query, case=False, regex=False, na=False
            )
            | filtered["_tags"].str.contains(
                search_query, case=False, regex=False, na=False
            )
        )
        filtered = filtered[mask]

    display_df = pd.DataFrame({
        "Feedback": filtered["_original"],
        "Cleaned text": filtered["_clean"],
        "Sentiment": filtered["_sentiment"],
        "Sentiment score": filtered["_score"],
        "Issue category": filtered["_category"],
        "Tags": filtered["_tags"],
    })

    st.write(f"**Matching records:** {len(display_df):,}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered feedback CSV",
        display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_railway_feedback.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Analyze a New Feedback Message")
    st.write(
        "Enter a new comment to estimate sentiment and identify candidate themes. "
        "These tags are generated by the dashboard's keyword rules and may differ "
        "from tags already stored in your CSV."
    )

    new_feedback = st.text_area(
        "Enter railway feedback",
        placeholder="Example: The train was delayed and the coach was not clean.",
        height=120,
    )

    if st.button("Analyze feedback", type="primary"):
        if not new_feedback.strip():
            st.warning("Please enter a feedback message first.")
        else:
            sentiment, score, tags, method = analyze_new_feedback(new_feedback)

            c1, c2 = st.columns(2)
            c1.metric("Estimated sentiment", sentiment)
            c2.metric(
                "Compound score",
                f"{score:.3f}" if score is not None else "Unavailable",
            )

            st.markdown("**Candidate tags**")
            for tag in tags:
                st.markdown(f"- `{tag}`")

            st.caption(method)
            st.caption(
                "This is an exploratory prediction, not a verified judgement. "
                "Keyword-based tags and lexicon sentiment can miss context, sarcasm, "
                "negation and spelling variations."
            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption(
    "Railway Opinion Mining | Python · pandas · scikit-learn · Streamlit · Plotly"
)
st.caption(
    "For academic interpretation: sentiment estimates and keyword-based tags "
    "should be reviewed manually. No classification accuracy is claimed here."
)

