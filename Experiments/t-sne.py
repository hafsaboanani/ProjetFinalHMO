from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def load_data_and_embeddings():
    # Chargement des données
    ng20 = fetch_20newsgroups(subset='test')
    corpus = ng20.data[:2000]
    labels = ng20.target[:2000]

    # Embedding
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(corpus)

    return embeddings, labels

def tsne_approach(embeddings, num_dimensions):
    # Appliquer t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=42)
    tsne_emb = tsne.fit_transform(embeddings)
    return tsne_emb

def kmeans_clustering(mat, k):
    # Normalisation des données
    scaler = StandardScaler()
    mat_normalized = scaler.fit_transform(mat)

    # Clustering avec K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    pred = kmeans.fit_predict(mat_normalized)
    return pred

def evaluate_clustering(pred, labels):
    # Évaluer les résultats du clustering
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)
    return nmi_score, ari_score

def main():
    embeddings, labels = load_data_and_embeddings()

    # Utilisation de l'approche t-SNE avec 3 composants principaux
    num_dimensions_tsne = 3  
    tsne_emb = tsne_approach(embeddings, num_dimensions_tsne)

    # Clustering avec K-means pour l'approche t-SNE
    pred_tsne = kmeans_clustering(tsne_emb, len(set(labels)))

    # Évaluation des résultats pour l'approche t-SNE
    nmi_score_tsne, ari_score_tsne = evaluate_clustering(pred_tsne, labels)
    print(f't-SNE - NMI: {nmi_score_tsne:.2f}, ARI: {ari_score_tsne:.2f}')

if __name__ == "__main__":
    main()
