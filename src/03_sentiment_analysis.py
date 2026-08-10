"""Create privacy-conscious sentiment features from patient message text.

The input data are expected to contain at least:
    - patient_id
    - full_text

If a date column is present, it is retained in the feature output.
Raw message text is never written to the output file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.models import TextClassifier
from tqdm import tqdm

REQUIRED_COLUMNS = {"patient_id", "full_text"}
DEFAULT_MAX_CHARS = 512
DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Flair sentiment features from patient message text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/grouped_text.csv"),
        help="Path to the private grouped message dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/patient_daily_sentiment.csv"),
        help="Path for the derived sentiment feature file.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum number of characters per Flair input chunk.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for Flair prediction.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    """Load message data and validate required columns."""
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Input file is missing required columns: {missing_list}")

    df = df.dropna(subset=["full_text"]).copy()
    df["full_text"] = df["full_text"].astype(str).str.strip()
    df = df[df["full_text"].ne("")].reset_index(drop=True)

    if df.empty:
        raise ValueError("No non-empty message text remains after cleaning.")

    return df


def split_text(text: str, max_chars: int) -> list[str]:
    """Split long text into fixed-width character chunks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def prepare_sentences(
    texts: pd.Series,
    max_chars: int,
) -> tuple[list[Sentence], list[int]]:
    """Convert message rows to Flair Sentence objects and track source rows."""
    sentences: list[Sentence] = []
    row_mapping: list[int] = []

    for row_idx, text in tqdm(
        enumerate(texts),
        total=len(texts),
        desc="Chunking text",
    ):
        for chunk in split_text(text, max_chars=max_chars):
            sentences.append(Sentence(chunk))
            row_mapping.append(row_idx)

    return sentences, row_mapping


def aggregate_predictions(
    df: pd.DataFrame,
    sentences: list[Sentence],
    row_mapping: list[int],
) -> pd.DataFrame:
    """Aggregate chunk-level predictions into one sentiment feature per input row.

    Flair returns a POSITIVE/NEGATIVE label and a confidence score for each chunk.
    The signed score is +confidence for positive chunks and -confidence for
    negative chunks. Row-level sentiment_score is the mean signed chunk score.
    """
    signed_scores: list[list[float]] = [[] for _ in range(len(df))]
    confidence_scores: list[list[float]] = [[] for _ in range(len(df))]

    for sentence, row_idx in tqdm(
        zip(sentences, row_mapping),
        total=len(sentences),
        desc="Aggregating predictions",
    ):
        label = sentence.labels[0]
        sign = 1.0 if label.value == "POSITIVE" else -1.0

        confidence_scores[row_idx].append(label.score)
        signed_scores[row_idx].append(sign * label.score)

    result = df.copy()
    result["confidence"] = [
        float(np.mean(values)) if values else np.nan
        for values in confidence_scores
    ]
    result["sentiment_score"] = [
        float(np.mean(values)) if values else np.nan
        for values in signed_scores
    ]
    result["sentiment"] = np.where(
        result["sentiment_score"] >= 0,
        "POSITIVE",
        "NEGATIVE",
    )

    return result


def select_safe_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only identifiers needed for modeling plus derived sentiment fields.

    The raw full_text column is intentionally excluded so patient message text is
    not copied into downstream feature files.
    """
    columns = ["patient_id"]

    if "date" in df.columns:
        columns.append("date")

    columns.extend(["sentiment", "confidence", "sentiment_score"])
    return df[columns].copy()


def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.input}")
    df = load_data(args.input)
    print(f"Loaded {len(df):,} message rows.")

    print("Loading Flair sentiment model...")
    classifier = TextClassifier.load("en-sentiment")

    sentences, row_mapping = prepare_sentences(
        df["full_text"],
        max_chars=args.max_chars,
    )
    print(f"Created {len(sentences):,} text chunks.")

    print("Running sentiment analysis...")
    classifier.predict(
        sentences,
        mini_batch_size=args.batch_size,
        verbose=True,
    )

    scored = aggregate_predictions(df, sentences, row_mapping)
    output = select_safe_output_columns(scored)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Saved sentiment features to: {args.output}")


if __name__ == "__main__":
    main()
