import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import io

# Set seaborn style
sns.set_style("whitegrid")

# Load sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_model()

# App layout
st.set_page_config(page_title="E-Consultation Sentiment Analyzer", layout="wide")
st.title("💬 E-Consultation Sentiment Analyzer")
st.markdown("Analyze sentiments from user feedback or consultation messages using AI-powered analysis.")

# Tabs
tab1, tab2 = st.tabs(["📤 Upload CSV", "✍️ Enter Text"])

all_results = []

# ============ TAB 1: CSV Upload ============ #
with tab1:
    st.header("📄 Upload Feedback Data")
    uploaded_file = st.file_uploader("Upload a CSV file containing feedback or messages", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        column = st.selectbox("🔎 Select the column containing feedback text", df.columns)

        with st.spinner("🔍 Analyzing sentiments..."):
            results = sentiment_analyzer(df[column].astype(str).tolist())
            df['Sentiment'] = [res['label'] for res in results]
            df['Score'] = [round(res['score'], 2) for res in results]
            all_results = df

        st.success("✅ Analysis complete!")

        with st.expander("🔢 View Raw Data with Sentiments", expanded=False):
            st.dataframe(df)

        # Sentiment Distribution
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(sentiment_counts)

        with col2:
            fig1, ax1 = plt.subplots()
            ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
            ax1.axis('equal')
            st.pyplot(fig1)

        # Word Cloud
        st.subheader("☁️ Word Cloud (All Feedback)")
        all_text = " ".join(df[column].astype(str))
        wordcloud = WordCloud(width=1000, height=500, background_color='white', max_words=200).generate(all_text)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wordcloud, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

        # Download Button
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Results as CSV", csv, file_name="sentiment_results.csv", mime="text/csv")

# ============ TAB 2: Manual Text Input ============ #
with tab2:
    st.header("✍️ Analyze Individual Message")
    text_input = st.text_area("Type or paste your feedback message below:")

    if st.button("🔍 Analyze Sentiment"):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                result = sentiment_analyzer(text_input)[0]
                sentiment = result['label']
                score = round(result['score'], 2)

            # Display result with emoji
            emoji = "😊" if sentiment == "POSITIVE" else ("😐" if sentiment == "NEUTRAL" else "😞")
            st.success(f"{emoji} **Sentiment:** {sentiment}")
            st.info(f"📈 **Confidence Score:** {score}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
