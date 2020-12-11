from inverted_index import *
from pagerank import *

# DELIVERY 1 #

# Indexing 500 documents from Dtest (core.py => docs_to_test = 500)
print("Building index...")
ix, ix_time, ix_space = indexing(corpus_directory, 2048, stemmed=stemming, d_test=True)
print("Done!")

print(f"Time to build index: {round(ix_time, 3)}s")
print(f"Disk space taken up by the index: {convert_filesize(ix_space)}")

# BM25 ranked query
print("Ranked query (using BM25) for topic R104 (p=20):")
print(ranking(104, 20, ix, "BM25"))

# DELIVERY 2 #

# Preprocessing 500 documents from Dtest (core.py => docs_to_test = 500)
corpus = process_documents(corpus_directory, stemmed=True, train=False)
topic_r104 = process_topic(104, topic_directory, stemmed=True)

# Calculating PageRank for the document graph made from the preprocessed documents
print("Top 20 PageRank docs with custom prior probabilities based on topic R104, edge weights and TF-IDF similarity:")
print(undirected_page_rank(topic_r104, corpus, n_docs=20, threshold=0.4,
                           sim="TF-IDF", use_priors=True, weighted=True))
