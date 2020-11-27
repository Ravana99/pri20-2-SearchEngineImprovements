import os
import re
from nltk.corpus import stopwords
from xml.etree import ElementTree
from sklearn.feature_extraction.text import TfidfVectorizer
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer


#######################################################################################################################

# Customize parameters here:

docs_to_train = None            # How many docs for training, set to None to vectorize all of the docs in D_train
docs_to_test = 500              # How many docs for testing, set to None to vectorize all of the docs in D_test
stemming = True
corpus_directory = os.path.join("..", "material", "rcv1")  # Directory of your rcv1 folder
topic_directory = os.path.join(corpus_directory, "..", "topics.txt")  # Directory of your topics.txt file
qrels_train_directory = os.path.join(corpus_directory, "..", "qrels.train")
qrels_test_directory = os.path.join(corpus_directory, "..", "qrels.test")
topic_ids = [104, 113, 121, 135, 146, 148, 159, 160, 175, 190]


#######################################################################################################################


# Preprocesses text
def preprocess_doc(doc, stemmed):
    if stemmed:
        analyzer = StemmingAnalyzer(stoplist=set(stopwords.words("english")))
    else:
        analyzer = StandardAnalyzer(stoplist=set(stopwords.words("english")))
    tokens = [token.text for token in analyzer(doc)]
    preprocessed_doc = ' '.join(tokens)
    return preprocessed_doc


# Returns list of pairs (doc_id, preprocessed_document)
def process_documents(corpus_dir, train=True, stemmed=True):
    n_docs = 0
    corpus = []

    if train:
        subdirs = filter(lambda x: x < "19961001" and x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus_dir))
    else:
        subdirs = filter(lambda x: x >= "19961001" and x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus_dir))

    docs_with_rels = get_docs_with_rels(qrels_train_directory, qrels_test_directory, train)

    for subdir in subdirs:
        for file in os.listdir(os.path.join(corpus_dir, subdir)):
            if int(file[:-10]) in docs_with_rels:
                doc_id, doc = extract_doc_content(os.path.join(corpus_dir, subdir, file))
                preprocessed_doc = preprocess_doc(doc, stemmed)
                corpus.append((doc_id, preprocessed_doc))
                n_docs += 1
                if (train and n_docs == docs_to_train) or (not train and n_docs == docs_to_test):
                    return corpus

    return corpus


def get_docs_with_rels(d_train, d_test, train):
    if train:
        rel_dir = d_train
    else:
        rel_dir = d_test

    with open(rel_dir) as f:
        res = {int(line.split()[1]) for line in f if int(line.split()[0][-3:]) in topic_ids}

    return res


# Returns a tuple (doc_id, preprocessed topic)
def process_topic(topic_id, topic_dir, stemmed=True):
    with open(topic_dir) as f:
        topics = f.read().split("</top>")[:-1]
    processed_topic = re.sub("<num> Number: R[0-9][0-9][0-9]", "", topics[topic_id-101])
    for tag in ("<top>", "<title>", "<desc> Description:", "<narr> Narrative:"):
        processed_topic = processed_topic.replace(tag, "")
    preprocessed_topic = (topic_id, preprocess_doc(processed_topic, stemmed).replace("documents ", "")
                                                                            .replace("document ", "")
                                                                            .replace("relevant ", "")
                                                                            .replace("relev ", "")
                                                                            .replace("irrelevant ", "")
                                                                            .replace("irrelev ", ""))
    return preprocessed_topic


# Returns list of tuples (doc_id, preprocessed topic)
def process_topics(topic_dir, stemmed=True):
    with open(topic_dir) as f:
        topics = f.read().split("</top>")[:-1]
    processed_topics = []
    for topic in topics:
        processed_topic = re.sub("<num> Number: R[0-9][0-9][0-9]", "", topic)
        for tag in ("<top>", "<title>", "<desc> Description:", "<narr> Narrative:"):
            processed_topic = processed_topic.replace(tag, "")
        processed_topics.append(processed_topic)
    preprocessed_topics = [(i+101, preprocess_doc(topic, stemmed).replace("documents ", "")
                                                                 .replace("document ", "")
                                                                 .replace("relevant ", "")
                                                                 .replace("relev ", "")
                                                                 .replace("irrelevant ", "")
                                                                 .replace("irrelev ", ""))
                           for i, topic in enumerate(processed_topics)]
    return preprocessed_topics


# Traverses document XML tree and extracts relevant fields
def extract_doc_content(file):
    tree = ElementTree.parse(file)
    root = tree.getroot()  # Root is <newsitem>
    doc_id = int(root.attrib["itemid"])
    res = ""
    for child in root:
        if child.tag in ("headline", "dateline", "byline"):  # Just extract text
            res += (child.text + " " if child.text is not None else "")
        elif child.tag == "text":  # Traverse all <p> tags and extract text from each one
            for paragraph in child:
                res += paragraph.text + " "
    return doc_id, res


# Returns list of lists (one for each topic), where each one of those lists stores tuples (doc_id, relevance in topic)
def extract_relevance(rel_dir):
    rels = [[] for _ in range(100)]
    with open(rel_dir) as f:
        for line in f:
            lst = line.split()
            rels[int(lst[0][-3:])-101].append((int(lst[1]), int(lst[2])))
    return rels


def vectorize_corpus(corpus):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(el[1] for el in corpus)
    # print(f"Number of docs and number of terms: {matrix.shape}")

    return vectorizer, matrix


def remove_duplicates_doc(document):
    doc = document.split()
    no_dups = []
    for word in doc:
        if document.count(word) > 1 and (word not in no_dups) or document.count(word) == 1:
            no_dups.append(word)
    return ' '.join(no_dups)


def remove_duplicates_corpus(corpus):
    no_dups_corpus = []
    for el in corpus:
        no_dups_doc = remove_duplicates_doc(el[1])
        no_dups_corpus.append((el[0], no_dups_doc))
    return no_dups_corpus


def main():
    corpus = process_documents(corpus_directory, train=True, stemmed=True)
    vectorizer, vectorized_corpus = vectorize_corpus(corpus)
    print("TF-IDF vectors for each document:")
    print(vectorized_corpus)
    print("Processed topics:")
    print(process_topics(topic_directory))


if __name__ == "__main__":
    main()
