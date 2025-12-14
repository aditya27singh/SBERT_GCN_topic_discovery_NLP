import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SBERT + GCN Topic Discovery", layout="wide")

st.title("Enhancing Topic Coherence and Discovery in NLP Literature")
st.markdown("""
This application presents **precomputed results** from a research project
combining semantic (Sentence-BERT) and structural (GCN) embeddings
for topic discovery in NLP literature.
""")

st.header("Method Summary")
st.markdown("""
- SBERT used for semantic representation
- Citation graph modeled using GCN
- Fusion via weighted concatenation
- Clustering evaluated using multiple metrics
""")


# -----------------------------
# Load precomputed data
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("output/artifacts/final_clustered_papers_with_graph.csv")
    with open("output/artifacts/cluster_keywords.json", "r") as f:
        keywords = json.load(f)
    return df, keywords

df, cluster_keywords = load_data()

# -----------------------------
# Results table
# -----------------------------
st.header("Quantitative Results")

results_df = pd.DataFrame([
    ["SBERT-only", 0.0523, 4.09, 56.39],
    ["GCN-only", 0.4261, 0.89, 1131.39],
    ["Combined", 0.2838, 1.37, 480.94],
], columns=["Method", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"])

st.dataframe(results_df, width="stretch")

# -----------------------------
# Cluster summaries
# -----------------------------
st.header("Cluster Keyword Summaries")

for cluster_id, words in cluster_keywords.items():
    st.subheader(f"Cluster {cluster_id}")
    st.write(", ".join(words))

# -----------------------------
# Example papers
# -----------------------------
st.header("Representative Papers")

cluster_choice = st.selectbox("Select Cluster", sorted(df["cluster_combined"].unique()))
sample = df[df["cluster_combined"] == cluster_choice].head(10)

st.dataframe(sample[["title", "year", "citationCount"]], use_container_width=True)

st.header("Embedding Visualization")
st.image("output/plots/umap_combined_clusters.png", caption="UMAP Visualization of Combined SBERT + GCN Embeddings")

st.download_button(
    label="Download Clustered Papers (CSV)",
    data=df.to_csv(index=False),
    file_name="clustered_papers.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("ETH-safe deployment: all results are precomputed, no live model execution.")
