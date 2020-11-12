import numpy as np
from sklearn.cluster import AgglomerativeClustering

from tfidf import *


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


def main():
    # corpus = process_documents(corpus_directory)
    corpus = process_topics(topic_directory)
    clusters = clustering(corpus,
                          clustering_model=AgglomerativeClustering(n_clusters=35, linkage="average", affinity="cosine"))
    np.set_printoptions(threshold=6)
    print(clusters)


if __name__ == "__main__":
    main()
