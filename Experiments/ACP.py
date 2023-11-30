from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np

# Chargement des données
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]

# Embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# ACP
def acp_approach(embeddings, num_dimensions):
    pca = PCA(n_components=num_dimensions)
    red_emb = pca.fit_transform(embeddings)
    return red_emb

# Clustering
def perform_clustering(mat, k):
    pred = np.random.randint(k, size=len(corpus))
    return pred

# Utilisation de l'approche ACP
num_dimensions_acp = 20
red_emb_acp = acp_approach(embeddings, num_dimensions_acp)

# Clustering
pred_acp = perform_clustering(red_emb_acp, len(set(labels)))

# Évaluation des résultats
nmi_score_acp = normalized_mutual_info_score(pred_acp, labels)
ari_score_acp = adjusted_rand_score(pred_acp, labels)

# Affichage des résultats
print(f'ACP - NMI: {nmi_score_acp:.2f}, ARI: {ari_score_acp:.2f}')