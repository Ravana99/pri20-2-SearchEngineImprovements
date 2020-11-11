import os
from xml.etree import ElementTree
from sklearn.feature_extraction.text import TfidfVectorizer
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer


#######################################################################################################################

# Customize parameters here:

stemming = True
docs_to_vectorize = 1000        # How many docs to vectorize, set to None to add vectorize of the docs in the corpus
corpus_directory = os.path.join("..", "material", "rcv1")  # Directory of your rcv1 folder

#######################################################################################################################


def preprocess_doc(doc):
    if stemming:
        analyzer = StemmingAnalyzer()
    else:
        analyzer = StandardAnalyzer()
    tokens = [token.text for token in analyzer(doc)]
    preprocessed_doc = ' '.join(tokens)
    return preprocessed_doc


def process_corpus(corpus_dir, d_train=True, d_test=False, stemmed=True):
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
    print(f"Number of docs and number of terms: {matrix.shape}")

    return vectorizer, matrix


def query(vectorizer, query_string):  # Incomplete (only calculates TF-IDF vector and returns it), may not be necessary
    preprocessed_query = preprocess_doc(query_string)
    return vectorizer.transform([preprocessed_query])


def main():
    corpus = process_corpus(corpus_directory, d_train=True, d_test=False, stemmed=True)
    vectorizer, vectorized_corpus = vectorize_corpus(corpus)
    print("TF-IDF vectors for each document:")
    print(vectorized_corpus)
    print("Query:")
    print(query(vectorizer, "economy mexico bajej"))


if __name__ == "__main__":
    main()
