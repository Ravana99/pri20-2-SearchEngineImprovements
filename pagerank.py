from core import *
import numpy as np
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from collections import Counter

import matplotlib.pyplot as plt


np.set_printoptions(threshold=6)


def build_graph(corpus, threshold, use_idf):
    doc_ids = (el[0] for el in corpus)
    docs = (el[1] for el in corpus)

    new_to_old_ids = {i: corpus[i][0] for i in range(len(corpus))}

    vectorizer = TfidfVectorizer(use_idf=use_idf)
    tfidf = vectorizer.fit_transform(docs)
    pairwise_similarities = (tfidf * tfidf.transpose()).toarray().tolist()
    edges = []
    for i in range(len(pairwise_similarities)):
        for j in range(i+1, len(pairwise_similarities[i])):
            if pairwise_similarities[i][j] > threshold:
                edges.append((i, j))

    # graph = nx.DiGraph()
    # graph.add_nodes_from([0, 1, 2, 3])
    # graph.add_edges_from([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)])

    graph = nx.Graph()
    graph.add_nodes_from(doc_ids)
    graph.add_edges_from((new_to_old_ids[i], new_to_old_ids[j]) for i, j in edges)

    # nx.draw(graph, with_labels=True)
    # plt.show()

    return graph


def undirected_page_rank(topic, corpus, n_docs=100, use_idf=True, threshold=0.9,
                         max_iter=50, damping=0.15, personalization=None, weight=None):
    graph = build_graph(corpus, threshold, use_idf=use_idf)

    # "Undirected graphs will be converted to a directed graph with two directed edges for each undirected edge"
    page_rank = pagerank(graph, max_iter=50, alpha=1-damping, personalization=personalization, weight=weight)

    return Counter(page_rank).most_common(n_docs)


def main():
    corpus = process_documents(corpus_directory, stemmed=True, train=True)  # Stemmed documents
    # corpus = process_documents(corpus_directory, stemmed=False, train=True)  # Non stemmed documents
    topics = process_topics(topic_directory, stemmed=True)  # Stemmed topics
    # corpus = process_topics(topic_directory, stemmed=False)  # Non stemmed topics

    top_docs = undirected_page_rank(topics[1], corpus, n_docs=250)
    print(top_docs)


if __name__ == "__main__":
    main()
