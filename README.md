# Enhancing Topic Coherence and Discovery in NLP Literature  
### Using Combined Semantic and Graph Embeddings

## Overview

Transformer-based language models capture rich semantic relationships, but often lack explicit
structural organization when applied to large scientific corpora. This project investigates
how **semantic embeddings** and **citation-based structural embeddings** contribute to topic
discovery in NLP literature, and whether their combination improves clustering quality.

We compare three representations:
- **Sentence-BERT (SBERT)** semantic embeddings  
- **Graph Convolutional Network (GCN)** embeddings learned from a citation graph  
- A **combined semantic–structural representation** formed via weighted fusion  

The focus is on **quantitative clustering behavior**, **metric interpretation**, and the
challenges of multimodal embedding fusion.

## Research Questions

1. How effective are semantic embeddings alone for topic clustering in NLP literature?
2. Does citation structure provide stronger organization than semantic similarity?
3. What is the effect of combining semantic and structural embeddings?
4. How reliable are standard clustering metrics in multimodal embedding spaces?

## Methodology (Summary)

- ~1000 NLP papers collected using the Semantic Scholar Graph API  
- Text representation: *title + abstract*  
- Semantic embeddings generated using **Sentence-BERT (all-MiniLM-L6-v2)**  
- Citation graph constructed from intra-corpus references  
- A **GCN** trained on the citation graph using SBERT-derived cluster labels as pseudo-supervision  
- Semantic and structural embeddings aligned and fused via **α-weighted concatenation**  
- PCA applied before clustering  
- KMeans clustering with fixed `k = 2` for fair comparison  

## Results

### Quantitative Clustering Performance

| Representation | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Cluster Sizes |
|----------------|--------------|------------------|---------------------|---------------|
| SBERT-only     | 0.0523       | 4.09             | 56.39               | 596 / 404     |
| GCN-only       | 0.4261       | 0.89             | 1131.39             | 615 / 379     |
| Combined       | 0.2838       | 1.37             | 480.94              | 385 / 609     |

**Key observations:**
- SBERT embeddings exhibit weak geometric separability, reflecting semantic continuity rather
  than discrete topic boundaries.
- GCN embeddings achieve the strongest clustering across all metrics, highlighting the
  organizing role of citation structure.
- The combined representation yields intermediate performance, improving upon SBERT-only
  clustering but not surpassing the GCN-only baseline.

## Limitations

- GCN training relies on pseudo-supervised labels derived from SBERT clustering.
- Multimodal fusion is performed via weighted concatenation rather than a learned fusion method.
- Evaluation is based on unsupervised geometric clustering metrics.
- Citation graphs may be incomplete due to API constraints.

## Future Work

- Explore learned fusion strategies (e.g., projection networks or attention-based fusion).
- Incorporate human-in-the-loop evaluation of topic coherence.
- Investigate alternative graph structures such as co-authorship or temporal citation networks.
- Extend analysis to larger corpora and longitudinal topic evolution.

## Installation & Setup

### Clone the Repository
    bash
    git clone https://github.com/aditya27singh/sbert-gcn-topic-discovery-nlp.git
    cd sbert-gcn-topic-discovery-nlp

### Create a virtual environment
    bash
    python -m venv venv
##### Windows
      bash
      venv\Scripts\activate
#### MacOS/Linux
     bash
     source venv/bin/activate

### Install dependencies
    bash
    pip install -r requirements.txt
