# Live FIFA World Cup Sentiment Analysis

Live FIFA World Cup Sentiment Analysis is a Streamlit dashboard for classifying football and FIFA-related text as `positive`, `neutral`, or `negative`. The app loads a trained sentiment model and vectorizer, accepts a CSV upload, predicts sentiment for a selected text column, and turns the results into an interactive dashboard with charts, search, confidence scores, and CSV export.

## Features

- Upload any CSV file that contains FIFA, World Cup, football, or fan-reaction text.
- Select the column that contains the text to analyze.
- Predict sentiment using the saved machine learning model in `model/sentiment_model.pkl`.
- Classify each row as `positive`, `neutral`, or `negative`.
- Display sentiment totals, distribution charts, and top words by sentiment.
- Search inside analyzed posts and view sentiment metrics for matching rows.
- Export the analyzed dataset as a CSV file.
- Show prediction confidence when the loaded model supports probabilities.

## Model Performance

The sentiment classifier was evaluated on a test set of 16,295 samples.

**Accuracy:** `86.44%`

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Negative | 0.86 | 0.78 | 0.82 | 2,598 |
| Neutral | 0.84 | 0.86 | 0.85 | 6,226 |
| Positive | 0.89 | 0.90 | 0.89 | 7,471 |
| Macro average | 0.86 | 0.84 | 0.85 | 16,295 |
| Weighted average | 0.86 | 0.86 | 0.86 | 16,295 |

The model performs best on positive tweets, with an F1-score of `0.89`, and maintains balanced weighted performance across all three sentiment classes with a weighted F1-score of `0.86`.

## Model Pipeline

The training workflow is documented in `model/model.py.py`, which was generated from the original analysis notebook. The pipeline follows these main steps:

1. Load FIFA World Cup tweet datasets.
2. Keep English tweets and remove rows with missing tweet content.
3. Use the Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest` model to generate sentiment labels for unlabeled tweets.
4. Combine generated labels with an existing labeled FIFA tweet dataset.
5. Clean and preprocess tweet text by removing special characters, converting text to lowercase, removing stopwords, and applying Porter stemming.
6. Convert text into numeric features using TF-IDF vectorization.
7. Train a Logistic Regression classifier for three-class sentiment prediction.
8. Evaluate the model with accuracy, precision, recall, F1-score, and a confusion matrix.
9. Save the trained model and vectorizer as pickle files for dashboard inference.

## Project Structure

```text
.
|-- app.py                    # Streamlit dashboard and prediction workflow
|-- search_results.py         # Search and sentiment summary helper functions
|-- requirements.txt          # Minimal app dependencies
|-- pyproject.toml            # Full project environment configuration
|-- uv.lock                   # Locked dependency versions for uv users
|-- model/
|   |-- model.py.py           # Training and evaluation pipeline exported from notebook
|   |-- sentiment_model.pkl   # Saved sentiment classification model
|   `-- vectorizer.pkl        # Saved TF-IDF vectorizer
`-- README.md
```

## Requirements

- Python 3.11 or newer
- Streamlit
- pandas
- scikit-learn
- Plotly

The minimal dashboard dependencies are listed in `requirements.txt`. The broader experimentation and training environment is listed in `pyproject.toml`.

## Installation

Clone the repository and move into the project folder:

```bash
git clone <repository-url>
cd Live-Fifa-World-Cup-Sentiment-Analysis
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dashboard dependencies:

```bash
pip install -r requirements.txt
```

If you are using `uv`, you can install from the project configuration instead:

```bash
uv sync
```

## Running the App

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser, usually at:

```text
http://localhost:8501
```

## How to Use

1. Open the dashboard.
2. Upload a CSV file from the sidebar.
3. Select the column that contains the text or tweet content.
4. Click **Analyze Sentiment**.
5. Review the sentiment summary, charts, top words, and analyzed table.
6. Use the search box to filter analyzed posts by keyword.
7. Download the sentiment-labeled CSV from the sidebar.

## Input Data Format

The app accepts CSV files with at least one text column. The column name can be anything because the dashboard lets you choose the text column after upload.

Example:

```csv
tweet
"Argentina played beautifully tonight"
"The referee ruined the match"
"France and Morocco fans are waiting for kickoff"
```

After analysis, the app adds:

- `sentiment`: predicted sentiment label
- `confidence_score`: prediction confidence percentage, when supported by the model

## Saved Model Files

The app expects these files to exist:

```text
model/sentiment_model.pkl
model/vectorizer.pkl
```

If either file is missing or cannot be loaded, the Streamlit sidebar will show an error and sentiment analysis will be disabled until the files are restored.

## Main Files

### `app.py`

Contains the Streamlit dashboard. It loads the model assets, handles CSV upload, runs predictions, builds sentiment charts, calculates top words, supports keyword search, and exports the analyzed CSV.

### `search_results.py`

Contains reusable helper functions for sentiment normalization, keyword search, and summary metrics.

### `model/model.py.py`

Contains the model training and evaluation workflow exported from the original notebook.

## Results Summary

The final classifier achieved `86.44%` test accuracy. Positive sentiment had the strongest class-level score with `0.89` F1-score, while neutral and negative sentiment reached `0.85` and `0.82` F1-score respectively. The weighted average F1-score of `0.86` indicates that the classifier performs consistently across the full test set while accounting for class distribution.

## Notes

- The model is designed for FIFA, World Cup, football, and social media text. Predictions may be less reliable on unrelated domains.
- The dashboard does not retrain the model; it only loads the saved pickle files and performs inference.
- Uploaded CSV files are processed during the Streamlit session and can be exported after analysis.
