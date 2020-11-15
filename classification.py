import numpy as np
from sklearn.svm import LinearSVC

from tfidf import *


np.set_printoptions(threshold=6)


def training(topic_id, d_train, r_train, classifier=LinearSVC()):
    topic_id -= 101
    rels = r_train[topic_id]

    old_to_new_id = {el[0]: i for i, el in enumerate(d_train)}
    new_to_old_id = {i: el[0] for i, el in enumerate(d_train)}

    vectorizer, matrix = vectorize_corpus(el[1] for el in d_train)
    train_target = []
    j = 0
    for i in range(len(d_train)):
        if rels[j][0] == d_train[i][0]:
            train_target.append(rels[j][1])
            j += 1
        else:
            train_target.append(0)

    classifier.fit(X=matrix, y=train_target)

    return classifier


def classify():
    pass


def evaluate():
    pass


def main():
    corpus = process_documents(corpus_directory, d_train=True, d_test=False)  # Stemmed documents
    topics = process_topics(topic_directory)  # Stemmed topics
    # corpus = process_documents(corpus_directory, stemmed=False)  # Non stemmed documents
    # topics = process_topics(topic_directory, stemmed=False)  # Non stemmed topics

    rels = extract_relevance(qrels_train_directory)
    classifier = training(102, corpus, rels)


if __name__ == "__main__":
    main()
