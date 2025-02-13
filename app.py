import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
import matplotlib.pyplot as plt
import sys

# ================== DEPENDENCY CHECKS ================== #
try:
    import torch
except ImportError:
    st.error("Missing PyTorch. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.0.1"])
    st.experimental_rerun()

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    st.error("Missing transformers. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.30.2"])
    st.experimental_rerun()

try:
    import spacy
except ImportError:
    st.error("Missing spacy. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "spacy==3.7.4"])
    st.experimental_rerun()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Installing English model...")
    from spacy.cli import download
    download("en_core_web_sm")
    st.experimental_rerun()

# ================== MODEL SETUP ================== #
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_models()

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

# ================== EQUATION EXTRACTION ================== #
def extract_equation_params(text):
    doc = nlp(text.lower())
    numbers = [float(token.text) for token in doc if token.like_num]

    if len(numbers) < 4:
        return None

    demand_intercept, demand_slope, supply_intercept, supply_slope = numbers[:4]

    demand_slope = -abs(demand_slope)

    return demand_intercept, demand_slope, supply_intercept, supply_slope

def calculate_equilibrium(demand_intercept, demand_slope, supply_intercept, supply_slope):
    quantity_eq = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
    price_eq = demand_intercept + demand_slope * quantity_eq

    cs_eq = (demand_intercept - price_eq) * quantity_eq / 2
    ps_eq = (price_eq - supply_intercept) * quantity_eq / 2
    sw_eq = cs_eq + ps_eq

    return round(quantity_eq, 2), round(price_eq, 2), round(cs_eq, 2), round(ps_eq, 2), round(sw_eq, 2)

def plot_market(demand_intercept, demand_slope, supply_intercept, supply_slope, quantity_eq, price_eq, market_name):
    q_range = np.linspace(0, quantity_eq * 1.5, 100)
    demand_curve = demand_intercept + demand_slope * q_range
    supply_curve = supply_intercept + supply_slope * q_range

    plt.figure(figsize=(8, 6))
    plt.plot(q_range, demand_curve, label="Demand Curve", color="blue")
    plt.plot(q_range, supply_curve, label="Supply Curve", color="green")
    plt.axhline(y=price_eq, color='red', linestyle='--', label="Equilibrium Price")
    plt.axvline(x=quantity_eq, color='purple', linestyle='--', label="Equilibrium Quantity")
    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(f"Market Equilibrium: {market_name}")
    plt.legend()
    st.pyplot(plt)

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

st.header("⚖️ Market Equilibrium Solver")

user_query = st.text_area("Describe the market conditions (e.g., 'Demand has an intercept of 100 and slope of -2, Supply has an intercept of 20 and slope of 3.'):")

if st.button("Calculate Equilibrium"):
    params = extract_equation_params(user_query)
    if params:
        demand_intercept, demand_slope, supply_intercept, supply_slope = params

        # Compute Equilibrium
        quantity_eq, price_eq, cs_eq, ps_eq, sw_eq = calculate_equilibrium(
            demand_intercept, demand_slope, supply_intercept, supply_slope
        )

        # Display Results
        st.subheader("Equilibrium Results")
        st.write(f"**Equilibrium Quantity:** {quantity_eq}")
        st.write(f"**Equilibrium Price:** {price_eq}")
        st.write(f"**Consumer Surplus:** {cs_eq}")
        st.write(f"**Producer Surplus:** {ps_eq}")
        st.write(f"**Social Welfare:** {sw_eq}")

        # Plot the market
        plot_market(demand_intercept, demand_slope, supply_intercept, supply_slope, quantity_eq, price_eq, "Market Analysis")
    else:
        st.warning("Please provide at least 4 numerical values for the intercepts and slopes.")
