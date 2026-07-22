from agents.cleaning_agent import CleaningAgent
from agents.embedding_agent import EmbeddingAgent
from agents.sentiment_agent import SentimentAgent
from agents.theme_agent import ThemeAgent
from agents.trend_agent import TrendAgent
from agents.rag_agent import RAGAgent
from agents.qa_agent import QAAgent
from utils.helpers import load_sample_reviews


def run_smoke():
    df = load_sample_reviews()

    cleaning = CleaningAgent()
    cleaned = cleaning.clean_dataframe(df)
    assert not cleaned.empty

    sentiment = SentimentAgent()
    sframe = sentiment.analyze_dataframe(cleaned)
    assert "Sentiment" in sframe.columns

    embedding = EmbeddingAgent()
    embedding.index_reviews(sframe)

    rag = RAGAgent(embedding_agent=embedding)
    results = rag.retrieve("slow support", top_k=3)
    print("Retrieved:", results)

    qa = QAAgent(api_key=None)
    answer = qa.answer_question("What are the main complaints?", results)
    print("Answer:", answer)


if __name__ == '__main__':
    run_smoke()
