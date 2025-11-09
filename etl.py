"""ETL pipeline for essay scoring feature generation.

This script loads the raw essay, prompt, and rubric score datasets, engineers
additional features, and persists the resulting training set and a
corresponding specification file inside ``./feature_store``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)


# Regular expressions for reference counting.
APA_CITATION_RE = re.compile(r"\(([A-Z][A-Za-z]+(?: et al\.)?), \d{4}\)")
IEEE_CITATION_RE = re.compile(r"\[\d+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training features for essay scoring.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing essays.csv, prompts.csv, and rubric_scores.csv.",
    )
    parser.add_argument(
        "--feature-store-dir",
        type=Path,
        default=Path("feature_store"),
        help="Destination directory for the generated parquet file and metadata.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used to compute prompt/essay embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )
    return parser.parse_args()


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_datasets(data_dir: Path) -> pd.DataFrame:
    """Load raw CSV datasets and merge them into a single DataFrame."""

    essays_path = data_dir / "essays.csv"
    prompts_path = data_dir / "prompts.csv"
    rubric_path = data_dir / "rubric_scores.csv"

    LOGGER.info("Loading datasets from %s", data_dir)

    essays = pd.read_csv(essays_path)
    prompts = pd.read_csv(prompts_path)
    rubric = pd.read_csv(rubric_path)

    expected_essay_cols = {"essay_id", "prompt_id", "essay_text"}
    expected_prompt_cols = {"prompt_id", "prompt_text"}
    required_rubric_cols = {
        "essay_id",
        "thesis",
        "argument",
        "evidence",
        "organization",
        "language",
        "conventions",
    }

    missing_essay_cols = expected_essay_cols - set(essays.columns)
    missing_prompt_cols = expected_prompt_cols - set(prompts.columns)
    missing_rubric_cols = required_rubric_cols - set(rubric.columns)

    if missing_essay_cols:
        raise ValueError(f"essays.csv missing required columns: {missing_essay_cols}")
    if missing_prompt_cols:
        raise ValueError(f"prompts.csv missing required columns: {missing_prompt_cols}")
    if missing_rubric_cols:
        raise ValueError(f"rubric_scores.csv missing required columns: {missing_rubric_cols}")

    LOGGER.debug("Merging essay and prompt datasets")
    df = essays.merge(prompts, on="prompt_id", how="left")
    LOGGER.debug("Merging rubric scores")
    df = df.merge(rubric, on="essay_id", how="left")

    df["essay_text"] = df["essay_text"].fillna("")
    df["prompt_text"] = df["prompt_text"].fillna("")

    ordered_columns = [
        "essay_id",
        "prompt_text",
        "essay_text",
        "thesis",
        "argument",
        "evidence",
        "organization",
        "language",
        "conventions",
    ]
    df = df[ordered_columns]
    return df


def count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0

    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False
    for char in cleaned:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel

    if cleaned.endswith("e") and syllables > 1:
        syllables -= 1

    return max(1, syllables)


def flesch_kincaid_grade(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))

    words = re.findall(r"\b\w+\b", text)
    word_count = max(1, len(words))
    syllable_count = sum(count_syllables(word) for word in words)

    grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
    return float(grade)


def quote_ratio(text: str) -> float:
    if not text:
        return 0.0

    quoted_segments = re.findall(r'"(.*?)"', text)
    quoted_chars = sum(len(segment) for segment in quoted_segments)
    total_chars = len(text)

    if total_chars == 0:
        return 0.0
    return quoted_chars / total_chars


def reference_count(text: str) -> int:
    if not text:
        return 0
    apa_refs = len(APA_CITATION_RE.findall(text))
    ieee_refs = len(IEEE_CITATION_RE.findall(text))
    return apa_refs + ieee_refs


def average_paragraph_length(text: str) -> float:
    if not text:
        return 0.0
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return 0.0

    word_counts = [len(re.findall(r"\b\w+\b", paragraph)) for paragraph in paragraphs]
    return float(np.mean(word_counts))


def compute_prompt_cosine(df: pd.DataFrame, model_name: str, batch_size: int) -> np.ndarray:
    LOGGER.info("Loading SentenceTransformer model '%s'", model_name)

    # Enforce offline execution to avoid unintended downloads.
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = SentenceTransformer(model_name)

    if df.empty:
        LOGGER.warning("Input dataframe is empty; returning zero-length cosine array")
        return np.array([], dtype=float)

    prompt_embeddings: List[np.ndarray] = []
    essay_embeddings: List[np.ndarray] = []

    prompt_texts = df["prompt_text"].tolist()
    essay_texts = df["essay_text"].tolist()

    LOGGER.info("Encoding prompts and essays to compute cosine similarity")
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        prompt_batch = prompt_texts[start:end]
        essay_batch = essay_texts[start:end]
        prompt_embeds = model.encode(prompt_batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        essay_embeds = model.encode(essay_batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        prompt_embeddings.append(prompt_embeds)
        essay_embeddings.append(essay_embeds)

    prompt_matrix = np.vstack(prompt_embeddings)
    essay_matrix = np.vstack(essay_embeddings)
    cosine_scores = np.sum(prompt_matrix * essay_matrix, axis=1)
    return cosine_scores


def engineer_features(df: pd.DataFrame, model_name: str, batch_size: int) -> pd.DataFrame:
    df = df.copy()

    LOGGER.info("Engineering textual features")
    df["quote_ratio"] = df["essay_text"].apply(quote_ratio)
    df["ref_count"] = df["essay_text"].apply(reference_count)
    df["avg_paragraph_len"] = df["essay_text"].apply(average_paragraph_length)
    df["readability_fk"] = df["essay_text"].apply(flesch_kincaid_grade)

    df["prompt_cosine"] = compute_prompt_cosine(df, model_name, batch_size)

    return df


def save_outputs(df: pd.DataFrame, feature_store_dir: Path) -> None:
    feature_store_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = feature_store_dir / "train.parquet"
    spec_path = feature_store_dir / "spec.json"

    LOGGER.info("Saving feature dataset to %s", parquet_path)
    df.to_parquet(parquet_path, index=False)

    spec = {
        "target_columns": ["thesis", "argument", "evidence", "organization", "language", "conventions"],
        "feature_columns": [
            "prompt_text",
            "essay_text",
            "prompt_cosine",
            "quote_ratio",
            "ref_count",
            "avg_paragraph_len",
            "readability_fk",
        ],
        "primary_key": "essay_id",
    }

    LOGGER.info("Saving feature specification to %s", spec_path)
    with spec_path.open("w", encoding="utf-8") as fp:
        json.dump(spec, fp, indent=2)


def main() -> None:
    configure_logging()
    args = parse_args()

    LOGGER.info("Starting ETL pipeline")
    df = load_datasets(args.data_dir)
    df_with_features = engineer_features(df, args.model_name, args.batch_size)
    save_outputs(df_with_features, args.feature_store_dir)
    LOGGER.info("ETL pipeline completed successfully")


if __name__ == "__main__":
    main()

