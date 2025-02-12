import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Check for missing packages
try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("Missing required package: beautifulsoup4. Run `pip install beautifulsoup4`")
    sys.exit(1)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy English model missing. Run `python -m spacy download en_core_web_sm`")
    sys.exit(1)

# Cache expensive operations
@st.cache_data
def scrape_wikipedia(url):
    """Scrape text from a Wikipedia page."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
        return None

@st.cache_data
def preprocess(text):
    """Tokenize, lemmatize, and remove stopwords/punctuation."""
    doc = nlp(text)
    return ' '.join([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ])

@st.cache_data
def build_knowledge_base(text):
    """Preprocess text and build TF-IDF matrix."""
    sentences = [sent.text.strip() for sent in nlp(text).sents if len(sent.text.strip()) > 10]
    df = pd.DataFrame(sentences, columns=["text"])
    df["processed_text"] = df["text"].apply(preprocess)
    
    vectorizer = TfidfVectorizer()
    return {
        "tfidf_matrix": vectorizer.fit_transform(df["processed_text"]),
        "texts": df["text"].tolist(),
        "vectorizer": vectorizer
    }

def query_knowledge_base(query, knowledge_base):
    """Find the most relevant text for a query."""
    query_processed = preprocess(query)
    query_vector = knowledge_base["vectorizer"].transform([query_processed])
    similarities = cosine_similarity(query_vector, knowledge_base["tfidf_matrix"]).flatten()
    most_similar_idx = similarities.argmax()
    return knowledge_base["texts"][most_similar_idx], similarities[most_similar_idx]

# Streamlit UI
st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button { background-color: #4CAF50; color: white; }
    .stTextInput>div>div>input { font-size: 16px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# App layout
st.title("📚 Natural Resource Economics Study Assistant")
url = st.text_input("Wikipedia URL:", value="https://en.wikipedia.org/wiki/Natural_resource_economics")

if st.button("🔍 Build Knowledge Base"):
    if not url.startswith("https://en.wikipedia.org/"):
        st.warning("Please enter a valid Wikipedia URL")
    else:
        with st.spinner("Scraping and processing..."):
            text = scrape_wikipedia(url)
            if text:
                st.session_state.knowledge_base = build_knowledge_base(text)
                st.success("Knowledge base ready!")

if "knowledge_base" in st.session_state:
    query = st.text_input("Ask a question:")
    if query:
        result, score = query_knowledge_base(query, st.session_state.knowledge_base)
        st.subheader("Top Match")
        st.write(f"**Score:** {score:.2f}/1.00")
        st.write(result)
