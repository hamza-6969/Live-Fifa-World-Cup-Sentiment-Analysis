from pathlib import Path
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from search_results import normalize_sentiment, search_posts, sentiment_metrics


MODEL_PATH = Path("model/sentiment_model.pkl")
VECTORIZER_PATH = Path("model/vectorizer.pkl")

SENTIMENT_COLORS = {
    "positive": "#16a34a",
    "negative": "#dc2626",
    "neutral": "#64748b",
}


st.set_page_config(
    page_title="FIFA Sentiment Analysis Dashboard",
    page_icon=":soccer:",
    layout="wide",
)


@st.cache_resource
def load_model_assets():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None, "missing"

    try:
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)
        with VECTORIZER_PATH.open("rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer, None
    except Exception:
        return None, None, "load_error"


def build_download_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


def get_confidence_scores(model, transformed_text):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(transformed_text)
    return probabilities.max(axis=1)


def get_top_words_by_sentiment(dataframe, text_column, max_words=12):
    rows = []

    for sentiment in sorted(dataframe["sentiment"].map(normalize_sentiment).unique()):
        sentiment_text = (
            dataframe.loc[
                dataframe["sentiment"].map(normalize_sentiment) == sentiment,
                text_column,
            ]
            .fillna("")
            .astype(str)
        )

        if sentiment_text.str.strip().eq("").all():
            continue

        word_vectorizer = CountVectorizer(
            stop_words="english",
            max_features=max_words,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )

        try:
            word_matrix = word_vectorizer.fit_transform(sentiment_text)
        except ValueError:
            continue

        word_counts = word_matrix.sum(axis=0).A1
        words = word_vectorizer.get_feature_names_out()

        for word, count in sorted(zip(words, word_counts), key=lambda item: item[1], reverse=True):
            rows.append(
                {
                    "sentiment": sentiment,
                    "word": word,
                    "count": int(count),
                }
            )

    return pd.DataFrame(rows)


def metric_card(title, value, accent_color):
    if isinstance(value, (int, float)):
        display_value = f"{value:,}"
    else:
        display_value = value

    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color: {accent_color};">
            <span>{title}</span>
            <strong>{display_value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            padding: 1.4rem 1.6rem;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: linear-gradient(135deg, #0f172a 0%, #14532d 100%);
            color: #f8fafc;
            margin-bottom: 1.4rem;
        }

        .app-header h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 0.45rem 0;
            letter-spacing: 0;
        }

        .app-header p {
            color: #dbeafe;
            font-size: 1rem;
            margin: 0;
            max-width: 760px;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0.4rem 0 0.8rem 0;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-top: 4px solid #0f766e;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            min-height: 106px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .metric-card span {
            display: block;
            color: #64748b;
            font-size: 0.86rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
        }

        .metric-card strong {
            color: #0f172a;
            display: block;
            font-size: 2rem;
            line-height: 1;
        }

        div[data-testid="stSidebar"] {
            background: #f8fafc;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


model, vectorizer, model_error = load_model_assets()

if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None

if "source_file_name" not in st.session_state:
    st.session_state.source_file_name = "fifa_sentiment_results.csv"

if "analyzed_text_column" not in st.session_state:
    st.session_state.analyzed_text_column = None


st.markdown(
    """
    <div class="app-header">
        <h1>FIFA Sentiment Analysis Dashboard</h1>
        <p>
            Analyze FIFA and football-related text from uploaded CSV files, review model predictions,
            and export a clean sentiment-labeled dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Analysis Controls")

    if model_error == "missing":
        st.error(
            "Model files were not found. Place sentiment_model.pkl and vectorizer.pkl inside the model folder."
        )
        st.caption("Expected paths: model/sentiment_model.pkl and model/vectorizer.pkl")
    elif model_error == "load_error":
        st.error("The model files could not be loaded. Please check that the saved files are valid pickle files.")
    else:
        st.success("Model assets loaded")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    dataframe = None
    text_column = None

    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.session_state.source_file_name = uploaded_file.name.replace(".csv", "_sentiment.csv")
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV is empty. Please upload a file with rows and columns.")
        except Exception:
            st.error("The CSV could not be read. Please upload a valid comma-separated file.")

    if dataframe is not None and not dataframe.empty:
        text_columns = dataframe.columns.tolist()
        text_column = st.selectbox("Text column", text_columns)
    elif uploaded_file is not None and dataframe is not None:
        st.warning("This CSV has no data rows to analyze.")

    analyze_clicked = st.button(
        "Analyze Sentiment",
        type="primary",
        use_container_width=True,
        disabled=dataframe is None or dataframe.empty or model_error is not None,
    )

    if st.session_state.analyzed_df is not None:
        st.download_button(
            "Download analyzed CSV",
            data=build_download_csv(st.session_state.analyzed_df),
            file_name=st.session_state.source_file_name,
            mime="text/csv",
            use_container_width=True,
        )


if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin analyzing FIFA or football-related text.")
    st.stop()

if dataframe is None:
    st.stop()

if dataframe.empty:
    st.warning("The uploaded CSV is empty. Add data rows and upload the file again.")
    st.stop()

st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(dataframe.head(10), use_container_width=True)

if analyze_clicked:
    if not text_column or text_column not in dataframe.columns:
        st.error("Please select a valid text column before running sentiment analysis.")
    else:
        try:
            analyzed_df = dataframe.copy()
            text_values = analyzed_df[text_column].fillna("").astype(str)
            transformed_text = vectorizer.transform(text_values)
            predictions = model.predict(transformed_text)
            analyzed_df["sentiment"] = predictions
            confidence_scores = get_confidence_scores(model, transformed_text)
            if confidence_scores is not None:
                analyzed_df["confidence_score"] = (confidence_scores * 100).round(2)
            st.session_state.analyzed_df = analyzed_df
            st.session_state.analyzed_text_column = text_column
            st.success("Sentiment analysis completed successfully.")
        except Exception:
            st.error("Prediction failed. Please check that the selected column contains text similar to the training data.")


if st.session_state.analyzed_df is None:
    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)
    st.info("Choose a text column in the sidebar and click Analyze Sentiment to generate dashboard results.")
    st.stop()


results_df = st.session_state.analyzed_df.copy()
results_df["sentiment"] = results_df["sentiment"].astype(str)
summary_metrics = sentiment_metrics(results_df)
sentiment_counts = summary_metrics["sentiment_counts"]
total_rows = summary_metrics["total_posts"]
positive_count = summary_metrics["positive"]
negative_count = summary_metrics["negative"]
neutral_count = summary_metrics["neutral"]
average_confidence = summary_metrics["average_confidence"]

st.markdown('<div class="section-title">Sentiment Summary</div>', unsafe_allow_html=True)
metric_cols = st.columns(5)
with metric_cols[0]:
    metric_card("Total rows analyzed", total_rows, "#0f766e")
with metric_cols[1]:
    metric_card("Positive", positive_count, SENTIMENT_COLORS["positive"])
with metric_cols[2]:
    metric_card("Negative", negative_count, SENTIMENT_COLORS["negative"])
with metric_cols[3]:
    metric_card("Neutral", neutral_count, SENTIMENT_COLORS["neutral"])
with metric_cols[4]:
    if average_confidence is None:
        metric_card("Avg confidence", "N/A", "#2563eb")
    else:
        metric_card("Avg confidence", f"{average_confidence:.1f}%", "#2563eb")

chart_data = (
    sentiment_counts.rename_axis("sentiment")
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
bar_col, donut_col = st.columns([1.35, 1])

bar_chart = px.bar(
    chart_data,
    x="sentiment",
    y="count",
    color="sentiment",
    color_discrete_map=SENTIMENT_COLORS,
    text="count",
)
bar_chart.update_layout(
    height=390,
    margin=dict(l=10, r=10, t=20, b=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False,
    xaxis_title="Sentiment",
    yaxis_title="Number of rows",
    font=dict(color="#000000"),
    xaxis=dict(title_font=dict(color="#000000"), tickfont=dict(color="#000000")),
    yaxis=dict(title_font=dict(color="#000000"), tickfont=dict(color="#000000")),
)
bar_chart.update_traces(textposition="outside", marker_line_width=0, textfont_color="#000000")

donut_chart = px.pie(
    chart_data,
    names="sentiment",
    values="count",
    color="sentiment",
    color_discrete_map=SENTIMENT_COLORS,
    hole=0.58,
)
donut_chart.update_traces(textinfo="percent+label", textposition="inside", textfont_color="#000000")
donut_chart.update_layout(
    height=390,
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="white",
    showlegend=False,
    font=dict(color="#000000"),
)

with bar_col:
    st.plotly_chart(bar_chart, use_container_width=True)
with donut_col:
    st.plotly_chart(donut_chart, use_container_width=True)

if average_confidence is None:
    st.info("Confidence scores are unavailable because this model does not provide prediction probabilities.")

st.markdown('<div class="section-title">Search Inside Results</div>', unsafe_allow_html=True)
search_text_column = st.session_state.analyzed_text_column
search_term = st.text_input(
    "Search posts containing",
    placeholder="Enter a word to search analyzed posts",
)

if search_term.strip():
    matching_posts = search_posts(results_df, search_text_column, search_term)
    search_summary = sentiment_metrics(matching_posts)

    search_metric_cols = st.columns(5)
    with search_metric_cols[0]:
        metric_card("Matching posts", search_summary["total_posts"], "#0f766e")
    with search_metric_cols[1]:
        metric_card("Positive", search_summary["positive"], SENTIMENT_COLORS["positive"])
    with search_metric_cols[2]:
        metric_card("Negative", search_summary["negative"], SENTIMENT_COLORS["negative"])
    with search_metric_cols[3]:
        metric_card("Neutral", search_summary["neutral"], SENTIMENT_COLORS["neutral"])
    with search_metric_cols[4]:
        if search_summary["average_confidence"] is None:
            metric_card("Avg confidence", "N/A", "#2563eb")
        else:
            metric_card("Avg confidence", f"{search_summary['average_confidence']:.1f}%", "#2563eb")
elif search_text_column:
    st.caption("Enter a word to see the post count and sentiment metrics for matching analyzed posts.")

text_column_for_words = st.session_state.analyzed_text_column
if text_column_for_words and text_column_for_words in results_df.columns:
    top_words_df = get_top_words_by_sentiment(results_df, text_column_for_words)

    st.markdown('<div class="section-title">Top Words by Sentiment</div>', unsafe_allow_html=True)
    if top_words_df.empty:
        st.info("Top words could not be calculated for this dataset.")
    else:
        top_words_chart = px.bar(
            top_words_df,
            x="count",
            y="word",
            color="sentiment",
            facet_col="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            orientation="h",
        )
        top_words_chart.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=35, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            xaxis_title="Word count",
            yaxis_title="",
            font=dict(color="#000000"),
        )
        top_words_chart.update_xaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
        top_words_chart.update_yaxes(
            matches=None,
            categoryorder="total ascending",
            title_font=dict(color="#000000"),
            tickfont=dict(color="#000000"),
        )
        top_words_chart.update_traces(textfont_color="#000000")
        top_words_chart.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
        st.plotly_chart(top_words_chart, use_container_width=True)

st.markdown('<div class="section-title">Analyzed Results</div>', unsafe_allow_html=True)
st.dataframe(results_df, use_container_width=True)
