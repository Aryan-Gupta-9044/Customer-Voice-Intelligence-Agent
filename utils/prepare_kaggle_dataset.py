"""Convert a downloaded Kaggle review CSV into the schema this app expects:
Review, Product, Rating, Date.

Usage:
    python utils/prepare_kaggle_dataset.py \
        --input path/to/kaggle_file.csv \
        --output data/kaggle_reviews.csv \
        --review-col reviews.text \
        --product-col name \
        --rating-col reviews.rating \
        --date-col reviews.date \
        --sample 3000

Run with --list-columns first if you're not sure what a dataset's columns are called:
    python utils/prepare_kaggle_dataset.py --input path/to/kaggle_file.csv --list-columns

Built-in presets (use --preset instead of the four --*-col flags) for a couple of
well-known Kaggle datasets that already line up well with this app's schema:

  datafiniti-amazon   -> Datafiniti "Consumer Reviews of Amazon Products"
                          https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
  womens-ecommerce    -> "Women's E-Commerce Clothing Reviews"
                          https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

PRESETS = {
    "datafiniti-amazon": {
        "review": "reviews.text",
        "product": "name",
        "rating": "reviews.rating",
        "date": "reviews.date",
    },
    "womens-ecommerce": {
        "review": "Review Text",
        "product": "Class Name",
        "rating": "Rating",
        "date": None,  # this dataset has no review date column
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the raw Kaggle CSV")
    parser.add_argument("--output", default="data/kaggle_reviews.csv", help="Where to write the converted CSV")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Use a built-in column mapping")
    parser.add_argument("--review-col", help="Column containing the review text")
    parser.add_argument("--product-col", help="Column containing the product name")
    parser.add_argument("--rating-col", help="Column containing the numeric rating")
    parser.add_argument("--date-col", help="Column containing the review date")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N rows (recommended for demos / browser deployment)")
    parser.add_argument("--list-columns", action="store_true", help="Print the input file's column names and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_columns:
        columns = pd.read_csv(args.input, nrows=0).columns.tolist()
        print("\n".join(columns))
        return

    mapping = dict(PRESETS.get(args.preset, {})) if args.preset else {}
    mapping["review"] = args.review_col or mapping.get("review")
    mapping["product"] = args.product_col or mapping.get("product")
    mapping["rating"] = args.rating_col or mapping.get("rating")
    mapping["date"] = args.date_col or mapping.get("date")

    if not mapping.get("review") or not mapping.get("rating"):
        sys.exit("You must provide at least --review-col and --rating-col (or a --preset that includes them).")

    frame = pd.read_csv(args.input)

    output = pd.DataFrame()
    output["Review"] = frame[mapping["review"]]
    output["Product"] = frame[mapping["product"]] if mapping.get("product") in frame.columns else "Unknown"
    output["Rating"] = pd.to_numeric(frame[mapping["rating"]], errors="coerce")
    output["Date"] = pd.to_datetime(frame[mapping["date"]], errors="coerce").dt.strftime("%Y-%m-%d") if mapping.get("date") in frame.columns else ""

    output = output.dropna(subset=["Review", "Rating"])

    if args.sample and args.sample < len(output):
        output = output.sample(n=args.sample, random_state=42)

    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
