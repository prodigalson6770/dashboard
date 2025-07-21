import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
from transformers import pipeline
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Caching reduces loading time for the model.
@st.cache_resource
def get_sentiment_pipeline():
    # You may substitute models as needed for your application domain.
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = get_sentiment_pipeline()

st.title("News Sentiment Analyzer (Powered by Transformers)")

# User Inputs
api_key = st.text_input("NewsAPI API Key", type="password")
topics = st.text_input("Topics (comma-separated):", "Technology, AI")
from_date = st.date_input("From date", value=date.today() - timedelta(days=7))
to_date = st.date_input("To date", value=date.today())
num_articles = st.slider("Number of articles", min_value=5, max_value=50, value=10)

if st.button("Fetch News & Analyze Sentiment"):
    with st.spinner("Fetching articles..."):
        query = " OR ".join([t.strip() for t in topics.split(",") if t.strip()])
        url = (
            f"https://newsapi.org/v2/everything?q={query}"
            f"&from={from_date}&to={to_date}&sortBy=publishedAt&pageSize={num_articles}&apiKey={api_key}"
        )
        resp = requests.get(url).json()
        articles = resp.get("articles", [])

    if articles:
        df = pd.DataFrame([{
            "title": a.get("title", ""),
            "published_at": a.get("publishedAt", "")[:10],
            "description": a.get("description", ""),
            "url": a.get("url", "")
        } for a in articles])

        # Sentiment Analysis with Transformer
        st.info("Analyzing sentiment using DistilBERT...")
        sentiments = sentiment_pipeline(df["title"].fillna("").tolist())
        df["sentiment"] = [x["label"] for x in sentiments]
        df["score"] = [x["score"] for x in sentiments]

        st.success(f"Fetched and analyzed {len(df)} articles.")
        st.dataframe(df[["title", "published_at", "sentiment", "score"]])

        # Visualization: Sentiment Distribution Pie
        fig_pie = px.pie(df, names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig_pie)

        # Visualization: Sentiment Bar by Date
        df["published_at"] = pd.to_datetime(df["published_at"])
        trend = df.groupby([df["published_at"].dt.date, "sentiment"]).size().reset_index(name="count")
        fig_trend = px.bar(trend, x="published_at", y="count", color="sentiment",
                           title="Sentiment Trend Over Time", barmode="group")
        st.plotly_chart(fig_trend)

        # Word Cloud
        if df["title"].any():
            st.write("### Headlines Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df["title"].dropna()))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            st.pyplot(plt)

        # CSV Download Button
        csv = df.to_csv(index=False).encode()
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="news_sentiment.csv",
            mime="text/csv",
        )
    else:
        st.warning("No articles found or invalid API key.")

st.markdown("---")
st.markdown(
    """
    **Tips for Data Science Students:**
    - Try switching to other transformer models for finance or social media (e.g., FinBERT, RoBERTa).
    - Analyze sentiment over custom time ranges for trend analysis.
    - Use results as an input for further text mining (topics, clusters, etc.).
    """
)
