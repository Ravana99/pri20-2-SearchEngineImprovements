import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cosine, cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from tfidf import *


np.set_printoptions(threshold=6)


def clustering(corpus,  # TODO: change default values
               clustering_model=AgglomerativeClustering(n_clusters=35, linkage="average", affinity="cosine")):

    vectorizer, matrix = vectorize_corpus(corpus)
    clustering_model = clustering_model.fit(matrix.toarray())

    n_clusters = len(set(clustering_model.labels_))

    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(clustering_model.labels_):
        clusters[label].append(i)

    tfidf_vectors = [[] for _ in range(n_clusters)]
    for i, lst in enumerate(clusters):
        tfidf_vectors[i] = [matrix[doc_id, :] for doc_id in lst]

    centroids = [[] for _ in range(n_clusters)]  # Centroids calculated using the mean
    for i, lst in enumerate(tfidf_vectors):
        x = lst[0]
        for j in range(1, len(lst)):
            x = x + lst[j]
        x = x/len(clusters[i])
        centroids[i] = x.toarray()[0]

    return [(centroids[i], set(clusters[i])) for i in range(n_clusters)]


# Returns number of docs in cluster, doc ids, centroid, medoid, label, median
def interpret(cluster, corpus):
    centroid = cluster[0]
    doc_ids = list(cluster[1])
    doc_ids.sort()

    # Calculates centroid in new vector space
    docs = [doc for i, doc in enumerate(corpus) if i in doc_ids]
    vectorizer, matrix = vectorize_corpus(docs)
    tfidf_vectors = [matrix[i, :] for i in range(len(doc_ids))]
    vec = tfidf_vectors[0]
    for i in range(1, len(tfidf_vectors)):
        vec = vec + tfidf_vectors[i]
    vec = vec / len(tfidf_vectors)
    new_centroid = vec.toarray()[0]

    # Labels cluster using non-discriminative labeling
    top_3_term_indices = new_centroid.argsort()[-3:][::-1]
    top_3_terms = [word for i, word in enumerate(vectorizer.get_feature_names()) if i in top_3_term_indices]
    label = ' '.join(top_3_terms)

    # Calculates medoid using cosine distance
    medoid = 0
    for i in range(1, len(tfidf_vectors)):
        aux_arr = tfidf_vectors[i].toarray()[0]
        medoid_arr = tfidf_vectors[medoid].toarray()[0]
        if cosine(aux_arr, new_centroid) < cosine(medoid_arr, new_centroid):
            medoid = i

    # Calculates approximate multivariate geometric median of cluster through unconstrained
    # minimization using the BFGS method with cosine distance
    vectorizer, matrix = vectorize_corpus(corpus)
    points = np.array([matrix[i, :].toarray()[0] for i in range(len(corpus)) if i in doc_ids])

    def agg_distance(x):
        return cdist([x], points, metric="cosine").sum()

    median = minimize(agg_distance, centroid).x

    return len(doc_ids), doc_ids, centroid, doc_ids[medoid], label, median


# Calculates:
# - The Silhouette Coefficient, to evaluate cohesion and separation combined
# - The Variance Ratio Criterion, to evaluate cluster validity based on average intra and inter cluster sum of squares
# - The Davies-Bouldin index, to evaluate solely separation (values closer to zero indicate a better partition)
def evaluate(corpus,
             clustering_model=AgglomerativeClustering(n_clusters=35, linkage="average", affinity="cosine")):
    vectorizer, matrix = vectorize_corpus(corpus)
    clustering_model = clustering_model.fit(matrix.toarray())
    sil_score = silhouette_score(matrix.toarray(), clustering_model.labels_, metric="cosine")
    vrc = calinski_harabasz_score(matrix.toarray(), clustering_model.labels_)
    dbi = davies_bouldin_score(matrix.toarray(), clustering_model.labels_)

    return sil_score, vrc, dbi


def main():
    # corpus = process_documents(corpus_directory)  # Stemmed documents
    corpus = process_topics(topic_directory)  # Stemmed topics
    # corpus = process_documents(corpus_directory, stemmed=False)  # Non stemmed documents
    # corpus = process_topics(topic_directory, stemmed=False)  # Non stemmed topics

    clusters = clustering(corpus,
                          clustering_model=AgglomerativeClustering(n_clusters=35, linkage="average", affinity="cosine"))
    print(f"Clusters: {clusters}")

    n_docs, docs_in_cluster, centroid, medoid, label, median = interpret(clusters[0], corpus)
    print(f"Number of docs in cluster 0: {n_docs}")
    print(f"Docs in cluster 0: {docs_in_cluster}")
    print(f"Cluster 0 centroid: {centroid}")
    print(f"Cluster 0 medoid: {medoid}")
    print(f"Suggested label for cluster 0: {label}")
    print(f"Geometric median of cluster 0: {median}")

    sil_score, vrc, dbi = \
        evaluate(corpus, clustering_model=AgglomerativeClustering(n_clusters=35, linkage="average", affinity="cosine"))
    print(f"Silhouette coefficient: {sil_score}")
    print(f"Variance Ratio Criterion: {vrc}")
    print(f"Davies-Bouldin index: {dbi}")


if __name__ == "__main__":
    main()
