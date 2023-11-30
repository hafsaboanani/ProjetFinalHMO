from umap.umap_ import UMAP
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer 

# Chargement des données
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]

# Embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# UMAP
def umap_approach(embeddings, num_dimensions):
    umap_model = UMAP(n_components=num_dimensions)
    red_emb = umap_model.fit_transform(embeddings)
    return red_emb

# Clustering
def perform_clustering(embeddings, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    pred = kmeans_model.fit_predict(embeddings)
    return pred

# Utilisation de l'approche UMAP
num_dimensions_umap = 20
red_emb_umap = umap_approach(embeddings, num_dimensions_umap)

# Clustering
pred_umap = perform_clustering(red_emb_umap, len(set(labels)))

# Évaluation des résultats
nmi_score_umap = normalized_mutual_info_score(labels, pred_umap)
ari_score_umap = adjusted_rand_score(labels, pred_umap)

# Affichage des résultats
print(f'UMAP - NMI: {nmi_score_umap:.2f}, ARI: {ari_score_umap:.2f}')

