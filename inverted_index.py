import shutil
import time
import math
from collections import Counter

# from whoosh.index import open_dir
from sklearn.neighbors import KNeighborsClassifier
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import *
from whoosh.reading import TermNotFound
from whoosh.query import Every
from whoosh import scoring

from core import *
from classification import training, classify


# By default, the writer will have 1GB (1024 MB) as the maximum memory for the indexing pool
# However, the actual memory used will be higher than this value because of interpreter overhead (up to twice as much)
def indexing(corpus, ram_limit=1024, d_test=True, stemmed=True):
    start_time = time.time()

    if stemming:
        analyzer = StemmingAnalyzer(stoplist=set(stopwords.words("english")))
    else:
        analyzer = StandardAnalyzer(stoplist=set(stopwords.words("english")))

    schema = Schema(doc_id=NUMERIC(stored=True),
                    content=TEXT(analyzer=analyzer))

    index_dir = os.path.join("indexes", "docs")

    # Clear existing indexes/docs folder and make new one
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir)

    # Create index in indexes/docs folder
    ix = create_in(index_dir, schema)
    writer = ix.writer(limitmb=ram_limit)
    traverse_folders(writer, corpus, d_test=d_test)
    writer.commit()

    end_time = time.time()

    # Traverses all files in the indexes/docs folder to calculate disk space taken up by the index
    space = 0
    for subdir, dirs, files in os.walk(index_dir):
        space += sum(os.stat(os.path.join(index_dir, file)).st_size for file in files)

    return ix, end_time - start_time, space


# Traverses all sub-folders/files in the corpus and adds every document to the index
def traverse_folders(writer, corpus, d_test):
    n_docs = 0

    if d_test:
        subdirs = filter(lambda x: x >= "19961001" and x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus))
    else:
        subdirs = filter(lambda x: x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus))

    docs_with_rels = get_docs_with_rels(qrels_train_directory, qrels_test_directory, False)

    for subdir in subdirs:
        for file in os.listdir(os.path.join(corpus, subdir)):
            if int(file[:-10]) in docs_with_rels:
                doc_id, doc = extract_doc_content(os.path.join(corpus, subdir, file))
                writer.add_document(doc_id=doc_id, content=doc)
                n_docs += 1
                if n_docs == docs_to_test:
                    return


def extract_topic_query(topic_id, index, k):
    topic_id = int(topic_id)-101  # Normalize topic identifier to start at 0

    topic = process_topic(topic_id, topic_directory)[1]

    if stemming:
        schema = Schema(id=NUMERIC(stored=True), content=TEXT(analyzer=StemmingAnalyzer(stoplist=set(stopwords.words("english")))))
    else:
        schema = Schema(id=NUMERIC(stored=True), content=TEXT(StandardAnalyzer(stoplist=set(stopwords.words("english")))))

    topic_index_dir = os.path.join("indexes", "aux_topic")

    # Delete directory if it already exists and create a new one
    if os.path.exists(topic_index_dir):
        shutil.rmtree(topic_index_dir)
    os.makedirs(topic_index_dir)

    # Create auxiliary index with only 1 "document" (in reality, a topic)
    aux_index = create_in(topic_index_dir, schema)
    writer = aux_index.writer()
    writer.add_document(id=0, content=topic)
    writer.commit()

    with aux_index.searcher() as aux_searcher:
        # Dictionary of term frequencies in the TOPIC
        tf_dic = {word.decode("utf-8"): aux_searcher.frequency("content", word)
                  for word in aux_searcher.lexicon("content")
                  if word.decode("utf-8") not in ("document", "relev", "irrelev", "relevant", "irrelevant")}
        n_tokens_in_topic = sum(tf_dic.values())
        tf_dic = {word: freq/n_tokens_in_topic for word, freq in tf_dic.items()}

        with index.searcher() as searcher:
            # Dictionary of document frequencies of each term against the DOCUMENT INDEX
            results = searcher.search(Every(), limit=None)  # Returns every document
            n_docs = len(results)
            df_dic = {word: searcher.doc_frequency("content", word) for word in tf_dic}
            idf_dic = {word: math.log10(n_docs/(df+1)) for word, df in df_dic.items()}

    # Variation of TF-IDF, that uses topic tf and topics idf but also the idf against the corpus
    tfidfs = {key: tf_dic[key] * idf_dic[key] for key, value in df_dic.items() if value > 0}

    return list(tup[0] for tup in Counter(tfidfs).most_common(k))


def boolean_query(topic, k, index):
    words = extract_topic_query(topic, index, k)
    with index.searcher() as searcher:
        # Retrieve every document id
        results = searcher.search(Every(), limit=None)
        # Initialize dictionary that counts how many query terms each document contains
        occurrences = {r["doc_id"]: 0 for r in results}
        doc_ids = [r["doc_id"] for r in results]
        doc_ids.sort()
        for word in words:
            search_occurrences(searcher, occurrences, doc_ids, word)

        res = [doc_id for doc_id, occurrence in occurrences.items() if occurrence >= k - round(0.2*k)]
        res.sort()
        return res


def search_occurrences(searcher, occurrences, doc_ids, word):
    aux_occurrences = occurrences.copy()
    try:
        for doc_id in searcher.postings("content", word).all_ids():
            # Makes sure each entry is only incremented at the end once even if the term shows up in multiple fields
            occurrences[doc_ids[doc_id]] = aux_occurrences[doc_ids[doc_id]] + 1
    except TermNotFound:
        return


def ranking(topic_id, p, index, model="TF-IDF"):
    topic_id = int(topic_id)-101       # Normalize topic identifier to start at 0
    if model == "TF-IDF":
        weighting = scoring.TF_IDF()
    elif model == "BM25":
        weighting = scoring.BM25F()
    else:
        raise ValueError("Invalid scoring model: please use 'TF-IDF' or 'BM25'")

    topic = process_topic(topic_id, topic_directory)[1]

    if stemming:
        analyzer = StemmingAnalyzer(stoplist=set(stopwords.words("english")))
    else:
        analyzer = StandardAnalyzer(stoplist=set(stopwords.words("english")))

    tokens = [token.text for token in analyzer(topic)]
    string_query = ' '.join(tokens)
    with index.searcher(weighting=weighting) as searcher:
        q = QueryParser("content", index.schema, group=OrGroup).parse(string_query)
        results = searcher.search(q, limit=p)
        return [(r["doc_id"], round(r.score, 4)) for r in results]


# Prints the entire index for debugging and manual analysis purposes
def print_index(index):
    with index.searcher() as searcher:
        results = searcher.search(Every(), limit=None)
        doc_ids = [r["doc_id"] for r in results]
        doc_ids.sort()
        print(f"Index for field 'content':")
        for word in searcher.lexicon("content"):
            print(word.decode("utf-8") + ": ", end="")
            for doc in searcher.postings("content", word).all_ids():
                print(doc_ids[doc], end=" ")
            print()


def convert_filesize(size):
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
    i = 0
    while size // 1024 > 0 and i < 4:
        size /= 1024.0
        i += 1
    return str(round(size, 3)) + " " + suffixes[i]


def ranking_with_classifier(train_corpus, test_corpus, train_rels, topics, p, ix, alpha1=0.5, alpha2=0.5):
    lst = []
    for topic_id in topics:
        results = ranking(topic_id, p, ix, "TF-IDF")
        results = [(el[0], el[1] / results[0][1]) for el in results]
        new_corpus = []
        for id1 in (el[0] for el in results):
            for el in test_corpus:
                if el[0] == id1:
                    new_corpus.append((id1, el[1]))

        topic = process_topic(topic_id, topic_directory)
        model = training(topic, train_corpus, train_rels,
                         model=KNeighborsClassifier(n_neighbors=25, metric="euclidean"))
        # model = training(topic, train_corpus, train_rels, model=MultinomialNB(alpha=1.0))
        classes = [classify(new_corpus[i][1], topic, model) for i in range(len(new_corpus))]
        results = [(el[0], el[1] * alpha1 + classes[i] * alpha2) for i, el in enumerate(results)]
        results.sort(reverse=True, key=lambda x: x[1])
        lst.append(results)
    return lst


def main():
    ix, ix_time, ix_space = indexing(corpus_directory, 2048, stemmed=stemming)
    # ix, ix_time, ix_space = indexing(corpus_directory, 2048, stemmed=False)
    # ix, ix_time, ix_space = open_dir(os.path.join("indexes", "docs")), 0, 0
    # print("Whole index:"); print_index(ix)
    print(f"Time to build index: {round(ix_time, 3)}s")
    print(f"Disk space taken up by the index: {convert_filesize(ix_space)}")

    """
    with ix.searcher() as searcher:
        results = searcher.search(Every(), limit=None)  # Returns every document
        print(f"Number of indexed docs: {len(results)}")

    print("Boolean queries for topic 104 (k=3, 1 mismatch):")
    print(boolean_query(104, 3, ix))

    print("Ranked query (using TF-IDF) for topic 104 (p=20):")
    print(ranking(104, 20, ix, "TF-IDF"))

    print("Ranked query (using BM25) for topic 104 (p=20):")
    print(ranking(104, 20, ix, "BM25"))
    """

    train_corpus = process_documents(corpus_directory, train=True)  # Stemmed documents
    test_corpus = process_documents(corpus_directory, train=False)  # Stemmed documents
    train_rels = extract_relevance(qrels_train_directory)

    results = ranking_with_classifier(train_corpus, test_corpus, train_rels, topic_ids, 500, ix)
    print(results)


if __name__ == "__main__":
    main()
