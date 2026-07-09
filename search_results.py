import pandas as pd


def normalize_sentiment(value):
    return str(value).strip().lower()


def search_posts(dataframe, text_column, search_term):
    if dataframe is None or not text_column or text_column not in dataframe.columns:
        return pd.DataFrame()

    term = str(search_term).strip()
    if not term:
        return pd.DataFrame()

    text_values = dataframe[text_column].fillna("").astype(str)
    return dataframe.loc[text_values.str.contains(term, case=False, regex=False)].copy()


def sentiment_metrics(dataframe):
    empty_metrics = {
        "total_posts": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "average_confidence": None,
        "sentiment_counts": pd.Series(dtype="int64"),
    }

    if dataframe is None or dataframe.empty or "sentiment" not in dataframe.columns:
        return empty_metrics

    sentiment_counts = dataframe["sentiment"].map(normalize_sentiment).value_counts()
    average_confidence = None
    if "confidence_score" in dataframe.columns:
        average_confidence = float(dataframe["confidence_score"].mean())

    return {
        "total_posts": len(dataframe),
        "positive": int(sentiment_counts.get("positive", 0)),
        "negative": int(sentiment_counts.get("negative", 0)),
        "neutral": int(sentiment_counts.get("neutral", 0)),
        "average_confidence": average_confidence,
        "sentiment_counts": sentiment_counts,
    }
