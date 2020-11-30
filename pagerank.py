from core import *
from inverted_index import indexing, ranking
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from collections import Counter


def build_graph(corpus, use_idf, threshold):
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
                edges.append((new_to_old_ids[i], new_to_old_ids[j], pairwise_similarities[i][j]))

    graph = nx.Graph()
    graph.add_nodes_from(doc_ids)
    for i, j, k in edges:
        graph.add_edge(i, j, weight=k)

    return graph


def get_priors(graph, topic, prior_sim):
    ix = indexing(corpus_directory, 2048, stemmed=stemming)[0]
    ranks = ranking(topic[0], len(graph.nodes), ix, prior_sim)
    priors = {node: 0 for node in graph.nodes}
    for doc, score in ranks:
        priors[doc] = score
    return priors


def undirected_page_rank(topic, corpus, n_docs=100, sim="TF-IDF", threshold=0.4,
                         max_iter=50, damping=0.15, use_priors=False, weighted=False):

    if sim not in ("TF", "TF-IDF", "BM25"):
        raise ValueError("Invalid similarity criterion: please use 'TF', 'TF-IDF' or 'BM25'")

    use_idf = sim != "TF"

    graph = build_graph(corpus, use_idf=use_idf, threshold=threshold)

    priors = get_priors(graph, topic, sim) if use_priors else None
    weight = "weight" if weighted else None

    # "Undirected graphs will be converted to a directed graph with two directed edges for each undirected edge"
    page_rank = pagerank(graph, max_iter=max_iter, alpha=1-damping,
                         personalization=priors, weight=weight)

    return Counter(page_rank).most_common(n_docs)


def ranking_with_pagerank(corpus, topics, p, sim, ix, threshold, use_priors, weighted, alpha1=0.25, alpha2=0.75):
    lst = []
    for topic_id in (el[0] for el in topics if el[0] in topic_ids):
        rank_results = ranking(topic_id, p, ix, sim)
        rank_results = [(el[0], el[1] / rank_results[0][1]) for el in rank_results]

        pagerank_results = undirected_page_rank(topics[topic_id-101], corpus, n_docs=docs_to_test, sim=sim,
                                                threshold=threshold, use_priors=use_priors, weighted=weighted)
        pagerank_results = [(el[0], el[1] / pagerank_results[0][1]) for el in pagerank_results]

        scored_docs = {x: 0 for x in set([el[0] for el in rank_results] + [el[0] for el in pagerank_results])}

        for doc, res in rank_results:
            scored_docs[doc] += alpha1 * res
        for doc, res in pagerank_results:
            scored_docs[doc] += alpha2 * res

        results = [(doc, res) for doc, res in scored_docs.items()]
        results.sort(reverse=True, key=lambda x: x[1])

        lst.append(results)

    return lst


def main():
    corpus = process_documents(corpus_directory, stemmed=True, train=False)  # Stemmed documents
    topics = process_topics(topic_directory, stemmed=True)  # Stemmed topics

    print("Top 20 PageRank docs (threshold=0.9, max_iter=50, sim='TF-IDF', use_priors=False, weighted=True:")
    print(undirected_page_rank(topics[3], corpus, n_docs=docs_to_test, threshold=0.9,
                               sim="TF-IDF", use_priors=False, weighted=True)[:20])

    ix = indexing(corpus_directory, 2048, stemmed=stemming)[0]

    print("Ranking with the aid of the PageRank scores for all topics in topic_ids:")
    print(ranking_with_pagerank(corpus, topics, docs_to_test, "TF-IDF", ix, True, True, 0.5, 0.5))


if debug and __name__ == "__main__":
    main()
