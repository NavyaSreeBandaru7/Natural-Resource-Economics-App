import streamlit as st
import requests
import pandas as pd
import sys

# ================== DEPENDENCY CHECKS ================== #
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError:
    st.error("Missing transformers or torch. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers torch"])
    st.experimental_rerun()

# ================== BERT MODEL SETUP ================== #
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

# ================== TEXT PROCESSING ================== #
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

@st.cache_data
def get_wikipedia_content(url):
    try:
        page_title = url.split("/")[-1]
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
        st.error(f"Error: {str(e)}")
        return None

@st.cache_data
def build_knowledge_base(text):
    sentences = [sent.strip() for sent in text.split('.') if len(sent.strip()) > 10]
    df = pd.DataFrame(sentences, columns=["text"])
    df["embeddings"] = df["text"].apply(lambda x: get_sentence_embedding(x))
    return {
        "embeddings": torch.vstack(df["embeddings"].tolist()),
        "texts": df["text"].tolist()
    }

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).cpu().numpy()

def query_knowledge_base(query, knowledge_base):
    query_embedding = get_sentence_embedding(query)
    similarities = cosine_similarity(query_embedding, knowledge_base["embeddings"]).flatten()
    most_similar_idx = similarities.argmax()
    return knowledge_base["texts"][most_similar_idx], similarities[most_similar_idx]

# ================== STREAMLIT UI ================== #
st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")
st.title("📚 Natural Resource Economics Study Assistant")

url = st.text_input("Wikipedia URL:", value="https://en.wikipedia.org/wiki/Natural_resource_economics")

if st.button("🔍 Build Knowledge Base"):
    if not url.startswith("https://en.wikipedia.org/"):
        st.warning("Please enter a valid Wikipedia URL")
    else:
        with st.spinner("Processing..."):
            text = get_wikipedia_content(url)
            if text:
                st.session_state.knowledge_base = build_knowledge_base(text)
                st.success("Knowledge Base Ready!")

if "knowledge_base" in st.session_state:
    query = st.text_input("Ask a question:")
    if query:
        result, score = query_knowledge_base(query, st.session_state.knowledge_base)
        st.subheader("Top Result")
        st.write(f"**Relevance:** {score:.2f}/1.00")
        st.write(result)
