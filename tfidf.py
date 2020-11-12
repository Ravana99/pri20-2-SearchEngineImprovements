import os
import re
from nltk.corpus import stopwords
from xml.etree import ElementTree
from sklearn.feature_extraction.text import TfidfVectorizer
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer


#######################################################################################################################

# Customize parameters here:

stemming = True
docs_to_vectorize = 1000        # How many docs to vectorize, set to None to add vectorize of the docs in the corpus
corpus_directory = os.path.join("..", "material", "rcv1")  # Directory of your rcv1 folder
topic_directory = os.path.join(corpus_directory, "..", "topics.txt")  # Directory of your topics.txt file

#######################################################################################################################


def preprocess_doc(doc):
    if stemming:
        analyzer = StemmingAnalyzer(stoplist=set(stopwords.words("english")))
    else:
        analyzer = StandardAnalyzer(stoplist=set(stopwords.words("english")))
    tokens = [token.text for token in analyzer(doc)]
    preprocessed_doc = ' '.join(tokens)
    return preprocessed_doc


def process_documents(corpus_dir, d_train=True, d_test=False, stemmed=True):
    global stemming
    stemming = stemmed

    n_docs = 0
    corpus = []

    if d_test and d_train:
        subdirs = filter(lambda x: x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus_dir))
    elif d_test:
        subdirs = filter(lambda x: x >= "19961001" and x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus_dir))
    elif d_train:
        subdirs = filter(lambda x: x < "19961001" and x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus_dir))
    else:
        raise ValueError("At least either d_test or d_train must be True")

    for subdir in subdirs:
        for file in os.listdir(os.path.join(corpus_dir, subdir)):
            doc = extract_doc_content(os.path.join(corpus_dir, subdir, file))
            preprocessed_doc = preprocess_doc(doc)
            corpus.append(preprocessed_doc)
            n_docs += 1
            if n_docs == docs_to_vectorize:
                return corpus

    return corpus


def process_topics(topic_dir, stemmed=True):
    with open(topic_dir) as f:
        topics = f.read().split("</top>")[:-1]
    processed_topics = []
    for topic in topics:
        processed_topic = re.sub("<num> Number: R[0-9][0-9][0-9]", "", topic)
        for tag in ("<top>", "<title>", "<desc> Description:", "<narr> Narrative:"):
            processed_topic = processed_topic.replace(tag, "")
        processed_topics.append(processed_topic)
    preprocessed_topics = [preprocess_doc(topic).replace("documents ", "")
                                                .replace("document ", "")
                                                .replace("relevant ", "")
                                                .replace("relev ", "")
                                                .replace("irrelevant ", "")
                                                .replace("irrelev ", "")
                           for topic in processed_topics]
    return preprocessed_topics


def extract_doc_content(file):
    tree = ElementTree.parse(file)
    root = tree.getroot()  # Root is <newsitem>
    res = ""
    for child in root:
        if child.tag in ("headline", "dateline", "byline"):  # Just extract text
            res += (child.text + " " if child.text is not None else "")
        elif child.tag == "text":  # Traverse all <p> tags and extract text from each one
            for paragraph in child:
                res += paragraph.text + " "
    return res


def vectorize_corpus(corpus):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    # print(f"Number of docs and number of terms: {matrix.shape}")

    return vectorizer, matrix


def query(vectorizer, query_string):  # Incomplete (only calculates TF-IDF vector and returns it), may not be necessary
    preprocessed_query = preprocess_doc(query_string)
    return vectorizer.transform([preprocessed_query])


def main():
    corpus = process_documents(corpus_directory, d_train=True, d_test=False, stemmed=True)
    vectorizer, vectorized_corpus = vectorize_corpus(corpus)
    print("TF-IDF vectors for each document:")
    print(vectorized_corpus)
    print("Query:")
    print(query(vectorizer, "economy mexico bajej"))
    print("Processed topics:")
    print(process_topics(topic_directory))


if __name__ == "__main__":
    main()
