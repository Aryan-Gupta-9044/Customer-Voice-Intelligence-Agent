from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_data_path(filename: str) -> Path:
    """Return an absolute path under the data folder."""
    return DATA_DIR / filename


def load_sample_reviews() -> pd.DataFrame:
    """Load the bundled sample review dataset."""
    sample_path = get_data_path("sample_reviews.csv")
    if sample_path.exists():
        return pd.read_csv(sample_path)

    return pd.DataFrame(
        [
            {
                "Review": "The cooling performance of this fan is excellent and quiet.",
                "Product": "Fan",
                "Rating": 5,
                "Date": "2026-06-01",
            },
            {
                "Review": "The product stopped working after two weeks and customer support was slow.",
                "Product": "Fan",
                "Rating": 1,
                "Date": "2026-06-05",
            },
            {
                "Review": "Very satisfied with the build quality and energy efficiency.",
                "Product": "Fan",
                "Rating": 4,
                "Date": "2026-06-12",
            },
            {
                "Review": "The remote control is unreliable and the batteries drain quickly.",
                "Product": "AC",
                "Rating": 2,
                "Date": "2026-06-08",
            },
            {
                "Review": "Installation was easy and the cooling is strong.",
                "Product": "AC",
                "Rating": 5,
                "Date": "2026-06-15",
            },
            {
                "Review": "I love the design but the app keeps disconnecting from the device.",
                "Product": "AC",
                "Rating": 3,
                "Date": "2026-06-20",
            },
            {
                "Review": "The water heater temperature control is inconsistent and service is poor.",
                "Product": "Water Heater",
                "Rating": 2,
                "Date": "2026-06-18",
            },
            {
                "Review": "Excellent durability and the heating is very fast.",
                "Product": "Water Heater",
                "Rating": 5,
                "Date": "2026-06-25",
            },
            {
                "Review": "The packaging arrived damaged and the instructions were unclear.",
                "Product": "Water Heater",
                "Rating": 3,
                "Date": "2026-06-28",
            },
            {
                "Review": "Great sound quality but the buttons feel cheap.",
                "Product": "Speaker",
                "Rating": 4,
                "Date": "2026-07-01",
            },
            {
                "Review": "Battery life is disappointing and charging takes too long.",
                "Product": "Speaker",
                "Rating": 2,
                "Date": "2026-07-02",
            },
            {
                "Review": "The speaker is compact and the bass is impressive.",
                "Product": "Speaker",
                "Rating": 5,
                "Date": "2026-07-05",
            },
        ]
    )


def build_review_id(row: Dict[str, Any]) -> str:
    """Create a stable identifier for a review row."""
    text = f"{row.get('Review', '')}|{row.get('Product', '')}|{row.get('Date', '')}"
    return hashlib.md5(text.encode("utf-8")).hexdigest()


COLUMN_ALIASES: Dict[str, list[str]] = {
    "Review": ["review", "reviews.text", "review text", "review_text", "text", "comment", "body", "review_body", "reviewtext"],
    "Product": ["product", "name", "product name", "product_name", "productname", "item", "title", "asin", "class name", "class_name"],
    "Rating": ["rating", "reviews.rating", "star_rating", "stars", "score", "review_rating", "overall"],
    "Date": ["date", "reviews.date", "review_date", "reviewdate", "date_added", "reviews.dateadded", "time", "review_time"],
}

REQUIRED_COLUMNS = ["Review", "Rating"]  # a dataset is unusable without at least these


class SchemaMappingError(ValueError):
    """Raised when an uploaded CSV's columns can't be matched to Review/Product/Rating/Date."""


def standardize_review_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename an uploaded CSV's columns onto this app's Review/Product/Rating/Date schema.

    Handles an exact match first, then falls back to a case/punctuation-insensitive alias
    lookup (e.g. Kaggle's "reviews.text" -> "Review", "reviews.rating" -> "Rating") so users
    can upload a raw Kaggle export without pre-processing it. If a required column truly can't
    be found, raises SchemaMappingError with the actual column names seen, rather than letting
    a later step fail with an opaque KeyError.
    """
    normalized_lookup = {str(col).strip().lower(): col for col in df.columns}

    rename_map: Dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue  # already exactly right
        for alias in aliases:
            if alias in normalized_lookup:
                rename_map[normalized_lookup[alias]] = target
                break

    result = df.rename(columns=rename_map)

    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise SchemaMappingError(
            "Couldn't find a "
            + " or ".join(missing)
            + f" column in this file. Columns found: {', '.join(str(c) for c in df.columns)}. "
            "Rename the relevant column(s) to 'Review'/'Product'/'Rating'/'Date', or run "
            "`python utils/prepare_kaggle_dataset.py --input <file> --list-columns` to map them."
        )

    for optional_column, default in (("Product", "Unknown"), ("Date", "")):
        if optional_column not in result.columns:
            result[optional_column] = default

    return result
