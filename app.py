import streamlit as st
import requests
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Check for missing packages
try:
    import spacy
except ImportError:
    st.error("Missing required package: spacy. Run `pip install spacy`")
    sys.exit(1)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy English model missing. Run `python -m spacy download en_core_web_sm`")
    sys.exit(1)

# Cache expensive operations
@st.cache_data
def get_wikipedia_content(url):
    """Get Wikipedia content using official API"""
    try:
        # Extract page title from URL
        page_title = url.split("/")[-1]
        
        # Use Wikipedia API
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts",
            "explaintext": True
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page["extract"]
        
    except Exception as e:
        st.error(f"Failed to fetch content: {str(e)}")
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

# App layout
st.title("📚 Natural Resource Economics Study Assistant")
url = st.text_input("Wikipedia URL:", value="https://en.wikipedia.org/wiki/Natural_resource_economics")

if st.button("🔍 Build Knowledge Base"):
    if not url.startswith("https://en.wikipedia.org/"):
        st.warning("Please enter a valid Wikipedia URL")
    else:
        with st.spinner("Fetching and processing content..."):
            text = get_wikipedia_content(url)
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
