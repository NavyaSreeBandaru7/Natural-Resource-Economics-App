import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Cache expensive operations
@st.cache_data
def scrape_wikipedia(url):
    """Scrape text from a Wikipedia page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

@st.cache_data
def preprocess(text):
    """Tokenize, lemmatize, and remove stopwords/punctuation."""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return ' '.join(tokens)

@st.cache_data
def build_knowledge_base(text):
    """Preprocess text and build TF-IDF matrix."""
    sentences = [sent.text.strip() for sent in nlp(text).sents if len(sent.text.strip()) > 10]
    df = pd.DataFrame(sentences, columns=["text"])
    df["processed_text"] = df["text"].apply(preprocess)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

    return {
        "tfidf_matrix": tfidf_matrix,
        "texts": df["text"].tolist(),
        "vectorizer": vectorizer
    }

def query_knowledge_base(query, knowledge_base):
    """Find the most relevant text for a query."""
    query_processed = preprocess(query)
    query_vector = knowledge_base["vectorizer"].transform([query_processed])

    cosine_similarities = cosine_similarity(
        query_vector, knowledge_base["tfidf_matrix"]
    ).flatten()

    most_similar_idx = cosine_similarities.argmax()
    return knowledge_base["texts"][most_similar_idx], cosine_similarities[most_similar_idx]

# Streamlit App
st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput input {
        font-size: 16px;
        padding: 10px;
    }
    .stMarkdown h1 {
        color: #2E86C1;
    }
    .stMarkdown h2 {
        color: #1A5276;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("📚 Study Assistant: Natural Resource Economics")
st.markdown("""
    Welcome to the Study Assistant! This tool helps you explore topics in **Natural Resource Economics** by scraping Wikipedia content and answering your questions.
    """)

# Sidebar for Instructions
with st.sidebar:
    st.header("📘 Instructions")
    st.markdown("""
    1. Enter a Wikipedia URL (e.g., `https://en.wikipedia.org/wiki/Natural_resource_economics`).
    2. Click **Scrape and Build Knowledge Base** to load the content.
    3. Enter your question in the query box.
    4. Get the most relevant answer from the scraped content!
    """)
    st.markdown("---")
    st.markdown("**Made with ❤️ by [Your Name]**")

# Input URL
st.header("🔗 Step 1: Enter Wikipedia URL")
url = st.text_input("Paste the Wikipedia URL here:", "https://en.wikipedia.org/wiki/Natural_resource_economics")

# Scrape and Build Knowledge Base
if st.button("📂 Scrape and Build Knowledge Base"):
    with st.spinner("⏳ Scraping Wikipedia... This may take a moment."):
        try:
            scraped_text = scrape_wikipedia(url)
            st.success("✅ Scraping complete!")
        except Exception as e:
            st.error(f"❌ Error scraping the URL: {e}")
            st.stop()

    with st.spinner("⏳ Building Knowledge Base..."):
        try:
            knowledge_base = build_knowledge_base(scraped_text)
            st.session_state.knowledge_base = knowledge_base
            st.success("✅ Knowledge Base built!")
        except Exception as e:
            st.error(f"❌ Error building the knowledge base: {e}")
            st.stop()

# Query the Knowledge Base
if "knowledge_base" in st.session_state:
    st.header("❓ Step 2: Ask a Question")
    query = st.text_input("Enter your question here:", placeholder="e.g., What is sustainable management?")

    if st.button("🔍 Search"):
        if query:
            with st.spinner("⏳ Searching for the best answer..."):
                try:
                    result, similarity = query_knowledge_base(query, st.session_state.knowledge_base)
                    st.markdown("---")
                    st.subheader("📖 Most Relevant Text:")
                    st.write(result)
                    st.markdown(f"**🔢 Similarity Score:** {similarity:.2f}")
                except Exception as e:
                    st.error(f"❌ Error processing your query: {e}")
        else:
            st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
st.markdown("### 🎓 Happy Studying!")
st.markdown("This tool is designed to help students explore and understand complex topics in Natural Resource Economics.")
