from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.cleaning_agent import CleaningAgent
from agents.embedding_agent import EmbeddingAgent
from agents.qa_agent import QAAgent
from agents.rag_agent import RAGAgent
from agents.sentiment_agent import SentimentAgent
from agents.theme_agent import ThemeAgent
from agents.trend_agent import TrendAgent
from utils.helpers import SchemaMappingError, load_sample_reviews, standardize_review_schema

st.set_page_config(page_title="Customer Voice Intelligence Agent", page_icon="📊", layout="wide")

SENTIMENT_COLORS = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}
# Large CSVs (tens of thousands of rows - common with Kaggle exports) are slow to embed/index
# with the fallback (no sentence-transformers) pipeline. Cap and sample rather than hang.
MAX_ROWS_DEFAULT = 5000

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: #ffffff0d;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 1rem 1rem 0.6rem 1rem;
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; opacity: 0.85; }
    .cva-hero {
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(120deg, rgba(59,130,246,0.15), rgba(34,197,94,0.10));
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 1.2rem;
    }
    .cva-hero h1 { margin-bottom: 0.15rem; }
    .cva-pill {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
</style>
"""


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file: Optional[Any] = None, max_rows: int = MAX_ROWS_DEFAULT) -> tuple[pd.DataFrame, bool]:
    """Load either an uploaded CSV or the built-in sample dataset.

    Column names are auto-mapped onto Review/Product/Rating/Date (handling common variants
    like Kaggle's "reviews.text"/"reviews.rating") so a raw downloaded CSV can be uploaded
    directly. Raises SchemaMappingError with the actual columns found if that's not possible.
    Returns (dataframe, was_sampled) - large files are randomly sampled down to max_rows so
    the fallback (no sentence-transformers) pipeline stays responsive.
    """
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        raw = load_sample_reviews()

    standardized = standardize_review_schema(raw)

    was_sampled = False
    if max_rows and len(standardized) > max_rows:
        standardized = standardized.sample(n=max_rows, random_state=42).reset_index(drop=True)
        was_sampled = True

    return standardized, was_sampled


def run_analysis(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Execute the full end-to-end analysis pipeline."""
    cleaning_agent = CleaningAgent()
    cleaned_frame = cleaning_agent.clean_dataframe(dataframe)

    sentiment_agent = SentimentAgent()
    sentiment_frame = sentiment_agent.analyze_dataframe(cleaned_frame)

    embedding_agent = EmbeddingAgent()
    embedding_agent.index_reviews(sentiment_frame)

    rag_agent = RAGAgent(embedding_agent=embedding_agent)
    theme_agent = ThemeAgent()
    trend_agent = TrendAgent()
    qa_agent = QAAgent(api_key=os.getenv("GEMINI_API_KEY"))

    themes = theme_agent.extract_themes(sentiment_frame)
    trend_summary = trend_agent.analyze_trends(sentiment_frame)

    return {
        "cleaned": cleaned_frame,
        "sentiment": sentiment_frame,
        "themes": themes,
        "trend_summary": trend_summary,
        "rag_agent": rag_agent,
        "qa_agent": qa_agent,
    }


def analyze_and_store(uploaded_file: Optional[Any], source_label: str, max_rows: int = MAX_ROWS_DEFAULT) -> bool:
    """Run the pipeline once and cache results + metadata in session state.

    Returns True on success. On failure, shows a clear st.error() (with the actual reason,
    not a generic message) and leaves any previously working analysis in place rather than
    crashing the whole page.
    """
    try:
        with st.spinner("Running cleaning, sentiment, embedding, theme and trend agents..."):
            dataframe, was_sampled = load_dataset(uploaded_file, max_rows)
            results = run_analysis(dataframe)
    except SchemaMappingError as exc:
        st.error(f"Couldn't analyze this file: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 - surface it to the user instead of an unhandled crash
        st.error(f"Analysis failed: {exc}")
        return False

    st.session_state["analysis_results"] = results
    st.session_state["data_source"] = source_label
    st.session_state["row_count"] = len(dataframe)
    st.session_state["was_sampled"] = was_sampled
    # Any evidence/answer computed against the previous dataset is now stale.
    st.session_state.pop("retrieved_reviews", None)
    st.session_state.pop("last_answer", None)
    return True


def render_hero(source_label: str, row_count: int) -> None:
    st.markdown(
        f"""
        <div class="cva-hero">
            <h1>📊 Customer Voice Intelligence Agent</h1>
            <p style="opacity:0.85; margin-bottom:0.5rem;">
                Evidence-backed review analysis for product and customer experience teams.
            </p>
            <span class="cva-pill">Source: {source_label}</span>
            <span class="cva-pill">{row_count} reviews loaded</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(sentiment_frame: pd.DataFrame) -> None:
    """Render KPI cards for review distribution."""
    counts = sentiment_frame["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    avg_rating = pd.to_numeric(sentiment_frame.get("Rating"), errors="coerce").mean()

    cols = st.columns(5)
    cards = [
        ("📝 Total Reviews", f"{len(sentiment_frame):,}"),
        ("🙂 Positive", int(counts.get("Positive", 0))),
        ("😐 Neutral", int(counts.get("Neutral", 0))),
        ("☹️ Negative", int(counts.get("Negative", 0))),
        ("⭐ Avg Rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A"),
    ]
    for column, (label, value) in zip(cols, cards):
        column.metric(label=label, value=value)


def render_sentiment_donut(sentiment_frame: pd.DataFrame) -> None:
    counts = sentiment_frame["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    figure = go.Figure(
        data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.55,
                marker=dict(colors=[SENTIMENT_COLORS.get(label, "#64748b") for label in counts.index]),
                sort=False,
            )
        ]
    )
    figure.update_layout(
        title="Sentiment Distribution",
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_product_breakdown(sentiment_frame: pd.DataFrame) -> None:
    if "Product" not in sentiment_frame.columns or sentiment_frame["Product"].eq("").all():
        st.info("No product column found - upload a CSV with a Product column to see this breakdown.")
        return

    grouped = sentiment_frame.groupby(["Product", "Sentiment"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)

    figure = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        figure.add_trace(
            go.Bar(
                x=grouped.index,
                y=grouped.get(sentiment, 0),
                name=sentiment,
                marker_color=SENTIMENT_COLORS[sentiment],
            )
        )
    figure.update_layout(
        title="Sentiment by Product",
        barmode="stack",
        margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_themes(themes: list[dict[str, Any]]) -> None:
    if not themes:
        st.info("No themes were extracted from the available reviews.")
        return

    theme_df = pd.DataFrame(themes).rename(columns={"Frequency": "Mentions"})
    theme_df["Keywords"] = theme_df["Keywords"].apply(lambda words: ", ".join(words) if words else "-")
    theme_df["Representative Reviews"] = theme_df["Representative Reviews"].apply(
        lambda reviews: " | ".join(reviews) if reviews else "-"
    )

    st.dataframe(
        theme_df[["Topic", "Mentions", "Keywords", "Representative Reviews"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mentions": st.column_config.ProgressColumn(
                "Mentions", min_value=0, max_value=int(theme_df["Mentions"].max()), format="%d"
            ),
        },
    )

    figure = go.Figure(go.Bar(x=theme_df["Mentions"], y=theme_df["Topic"], orientation="h", marker_color="#3b82f6"))
    figure.update_layout(title="Theme Frequency", margin=dict(t=50, b=10, l=10, r=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(figure, use_container_width=True)


def render_trends(trend_summary: dict) -> None:
    charts = trend_summary.get("charts", {})
    has_dated_rows = bool(trend_summary.get("theme_frequency"))
    if not charts or not has_dated_rows:
        st.info("Not enough valid dates in the dataset to build a trend chart. Make sure your CSV has a Date column (YYYY-MM-DD).")
        return

    tab_labels = {"week": "By Week", "month": "By Month", "quarter": "By Quarter"}
    tabs = st.tabs([tab_labels.get(name, name.title()) for name in charts.keys()])
    for tab, (period_name, figure) in zip(tabs, charts.items()):
        with tab:
            for trace in figure.data:
                if trace.name in SENTIMENT_COLORS:
                    trace.marker.color = SENTIMENT_COLORS[trace.name]
            st.plotly_chart(figure, use_container_width=True)


def render_qa(results: Dict[str, Any]) -> None:
    question = st.text_input(
        "Ask a question about the reviews",
        placeholder="e.g. What are the main complaints about the AC product?",
    )
    ask_clicked = st.button("Get Evidence-Based Answer", type="primary")

    if ask_clicked:
        if question.strip():
            with st.spinner("Retrieving evidence and generating an answer..."):
                retrieved = results["rag_agent"].retrieve(question, top_k=5)
                response = results["qa_agent"].answer_question(question, retrieved)
            # Use the evidence the QA agent actually relied on (post relevance filtering),
            # not the raw top-k, so the Evidence Explorer matches what backs the answer.
            st.session_state["retrieved_reviews"] = response.get("Evidence") or retrieved
            st.session_state["last_answer"] = response
        else:
            st.warning("Please enter a question first.")

    last_answer = st.session_state.get("last_answer")
    if last_answer:
        st.markdown("**Answer**")
        st.write(last_answer["Answer"])
        confidence = last_answer.get("Confidence", 0.0)
        st.progress(min(1.0, max(0.0, confidence)), text=f"Confidence: {confidence:.2f}")
        if last_answer.get("Supporting Reviews"):
            st.markdown("**Supporting Reviews**")
            for snippet in last_answer["Supporting Reviews"]:
                st.write("- " + snippet)


def render_evidence_viewer(results: Dict[str, Any]) -> None:
    if "retrieved_reviews" in st.session_state and st.session_state["retrieved_reviews"]:
        st.caption("Showing evidence retrieved for your last question above.")
        evidence = st.session_state["retrieved_reviews"]
    else:
        st.caption("No question asked yet - showing a preview retrieval for 'customer complaints'.")
        evidence = results["rag_agent"].retrieve("customer complaints", top_k=5)

    if evidence:
        st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)
    else:
        st.info("No matching reviews were found.")


def main() -> None:
    """Render the main dashboard UI."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Upload Reviews")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        st.caption(
            "Needs a review-text column and a rating column. Product/Date are optional. "
            "Common alternate names (e.g. Kaggle's 'reviews.text', 'reviews.rating') are "
            "detected automatically."
        )
        max_rows = st.number_input(
            "Max rows to analyze",
            min_value=100,
            max_value=200_000,
            value=MAX_ROWS_DEFAULT,
            step=500,
            help="Large files are randomly sampled down to this many rows so analysis stays fast. Raise it if you want more coverage (slower).",
        )
        analyze_button = st.button("Analyze Reviews", type="primary", use_container_width=True)
        st.divider()
        if st.session_state.get("analysis_results"):
            sampled_note = " (randomly sampled)" if st.session_state.get("was_sampled") else ""
            st.success(f"Loaded {st.session_state.get('row_count', 0)} reviews{sampled_note} from {st.session_state.get('data_source', 'sample data')}.")

    # Run the pipeline exactly once on first load (using the bundled sample data) and again
    # only when the user explicitly clicks "Analyze Reviews" - NOT on every widget interaction
    # (e.g. typing a question), which would otherwise re-run the whole pipeline every rerun.
    if "analysis_results" not in st.session_state:
        analyze_and_store(None, "bundled sample data")

    if analyze_button:
        source_label = f"uploaded file '{uploaded_file.name}'" if uploaded_file is not None else "bundled sample data"
        analyze_and_store(uploaded_file, source_label, max_rows=int(max_rows))

    if "analysis_results" not in st.session_state:
        # The very first run failed (e.g. a broken bundled sample) - nothing to render.
        st.stop()

    results = st.session_state["analysis_results"]
    sentiment_frame = results["sentiment"]
    themes = results["themes"]
    trend_summary = results["trend_summary"]

    render_hero(st.session_state.get("data_source", "bundled sample data"), st.session_state.get("row_count", len(sentiment_frame)))
    render_metrics(sentiment_frame)

    overview_tab, themes_tab, trends_tab, qa_tab, evidence_tab = st.tabs(
        ["📈 Overview", "🧩 Themes", "📅 Trends", "💬 Ask a Question", "🔍 Evidence Explorer"]
    )

    with overview_tab:
        col1, col2 = st.columns(2)
        with col1:
            render_sentiment_donut(sentiment_frame)
        with col2:
            render_product_breakdown(sentiment_frame)

        with st.expander("View cleaned data"):
            st.dataframe(
                sentiment_frame[["Review", "Product", "Rating", "Date", "Sentiment", "Confidence"]],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download cleaned + scored data as CSV",
                data=sentiment_frame.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_reviews_with_sentiment.csv",
                mime="text/csv",
            )

    with themes_tab:
        render_themes(themes)

    with trends_tab:
        render_trends(trend_summary)

    with qa_tab:
        render_qa(results)

    with evidence_tab:
        render_evidence_viewer(results)


if __name__ == "__main__":
    main()
