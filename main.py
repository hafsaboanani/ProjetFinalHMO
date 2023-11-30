from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from umap.umap_ import UMAP

def acp_approach(embeddings, num_dimensions):
    pca = PCA(n_components=num_dimensions)
    red_emb = pca.fit_transform(embeddings)
    return red_emb

def umap_approach(embeddings, num_dimensions):
    umap_model = UMAP(n_components=num_dimensions)
    red_emb = umap_model.fit_transform(embeddings)
    return red_emb

def tsne_approach(embeddings, num_dimensions):
    tsne = TSNE(n_components=num_dimensions, random_state=42)
    tsne_emb = tsne.fit_transform(embeddings)
    return tsne_emb

def kmeans_clustering(mat, k):
    scaler = StandardScaler()
    mat_normalized = scaler.fit_transform(mat)
    kmeans = KMeans(n_clusters=k, random_state=42)
    pred = kmeans.fit_predict(mat_normalized)
    return pred

def evaluate_clustering(pred, labels):
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)
    return nmi_score, ari_score

ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)
k = len(set(labels))

methods = ['ACP', 'UMAP', 't-SNE']
for method in methods:
    if method == 'ACP':
        red_emb = acp_approach(embeddings, 20)
    elif method == 'UMAP':
        red_emb = umap_approach(embeddings, 20)
    elif method == 't-SNE':
        red_emb = tsne_approach(embeddings, 3)

    # Perform clustering
    pred = kmeans_clustering(red_emb, k)

    # Evaluate clustering results
    nmi_score, ari_score = evaluate_clustering(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')