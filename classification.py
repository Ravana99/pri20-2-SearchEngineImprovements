# from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from scipy.spatial.distance import cosine
from math import isnan
from numpy import seterr
import numpy as np

from tfidf import *


np.set_printoptions(threshold=200)


bin_prob_threshold = 0.2  # Value between 0 and 1, probability values >= this threshold will be regarded as relevant
all_features = True       # Run classification with all features (TF, IDF, TF-IDF, BM25) or just TF-IDF

tf_vectorizer = CountVectorizer()
idf_vectorizer = TfidfVectorizer()
tfidf_vectorizer = TfidfVectorizer()
bm25_vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
aux_bm25_vectorizer = CountVectorizer()

seterr(invalid="ignore")


def create_target(rels, corpus, max_values=None):
    if max_values is None:
        max_values = len(corpus)
    else:
        max_values = min(max_values, len(corpus))
    target = []
    j = 0
    for i in range(max_values):
        if j == len(rels):
            target.append(0)
            continue
        elif rels[j][0] == corpus[i][0]:
            target.append(rels[j][1])
            j += 1
        else:
            target.append(0)

    return target


avdl = 0


def training(topic, d_train, r_train, model):
    topic_id = topic[0] - 101
    rels = r_train[topic_id]

    global tf_vectorizer
    global idf_vectorizer
    global tfidf_vectorizer
    global bm25_vectorizer

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(el[1] for el in d_train)
    topic_tfidf_vec = tfidf_vectorizer.transform([topic[1]])
    doc_tfidf_vec = tfidf_vectorizer.transform(el[1] for el in d_train)
    cosine_distances = [cosine(doc_tfidf_vec[i].toarray(), topic_tfidf_vec[0].toarray())
                        for i in range(doc_tfidf_vec.shape[0])]
    tfidf_feature = np.array([[1 if isnan(cosine_distances[i]) else cosine_distances[i]
                               for i in range(doc_tfidf_vec.shape[0])]]).transpose()

    features = tfidf_feature

    if all_features:
        tf_vectorizer = CountVectorizer()
        tf_vectorizer.fit([topic[1]])
        tf_doc = tf_vectorizer.transform(el[1] for el in d_train)
        tf_feature = tf_doc.sum(axis=1)

        corpus_no_dups = remove_duplicates_corpus(d_train)
        idf_vectorizer = TfidfVectorizer()
        idf_vectorizer.fit(el[1] for el in corpus_no_dups)
        idf_doc = idf_vectorizer.transform(el[1] for el in corpus_no_dups)
        idf_feature = idf_doc.sum(axis=1)

        bm25_vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        corpus = [el[1] for el in d_train]
        bm25_fit(corpus)
        bm25_res = bm25_transform(corpus, topic[1])
        bm25_feature = np.array([[el for el in bm25_res]]).transpose()

        features = hstack([tf_feature, idf_feature, tfidf_feature, sparse.csc_matrix(np.array(bm25_feature))]).toarray()

    # Creates relevance list to be used to train the model
    train_target = create_target(rels, d_train)

    model.fit(X=features, y=train_target)

    return model


def bm25_fit(corpus):
    global bm25_vectorizer
    global avdl

    bm25_vectorizer.fit(corpus)

    # Generates a CountVectorizer using the information from our TfidfVectorizer
    aux_vectorizer = super(TfidfVectorizer, bm25_vectorizer)

    # Computes term frequencies for each document
    term_frequencies = aux_vectorizer.transform(corpus)

    # Computes number of terms per document
    n_terms = term_frequencies.sum(axis=1)

    # Computes average document length (average number of terms per document)
    avdl = n_terms.mean()


def bm25_transform(corpus, topic, b=0.75, k1=1.6):
    # Generates a CountVectorizer using the information from our TfidfVectorizer
    aux_vectorizer = super(TfidfVectorizer, bm25_vectorizer)

    # Computes term frequencies for each document
    term_frequencies = aux_vectorizer.transform(corpus)

    # Computes number of terms per document
    dl = np.asarray(term_frequencies.sum(axis=1)).transpose()[0]

    # Computes term frequencies for the topic given the vocabulary of the fitted corpus
    topic = aux_vectorizer.transform([topic])

    # Considers only the terms that are present in the topic
    term_frequencies = term_frequencies[:, topic.nonzero()[1]]

    # Retrieve log idf for each term in the topic
    idf = bm25_vectorizer._tfidf.idf_[topic.indices] - 1.0
    idf = np.tile(idf, (term_frequencies.shape[0], 1))

    # Numerator (including log idf term)
    numerator = term_frequencies.multiply(idf) * (k1 + 1)

    # For each of the tf vectors, add it to the other part of the BM25 denominator
    denominator = term_frequencies + (k1 * ((1 - b) + b * (dl / avdl)))[:, np.newaxis]

    # Apply the formula and return the column as an array
    return np.asarray((numerator / denominator).sum(axis=1)).transpose()[0]


def classify(doc, topic, model):
    tfidf_doc = tfidf_vectorizer.transform([doc])
    tfidf_topic = tfidf_vectorizer.transform([topic[1]])
    cosine_distance = cosine(tfidf_doc[0].toarray(), tfidf_topic[0].toarray())
    tfidf_feature = np.array([[1 if isnan(cosine_distance) else cosine_distance]])

    test_features = tfidf_feature

    if all_features:
        tf_doc = tf_vectorizer.transform([doc])
        tf_feature = tf_doc.sum(axis=1)

        doc_no_dups = remove_duplicates_doc(doc)
        idf_doc = idf_vectorizer.transform([doc_no_dups])
        idf_feature = idf_doc.sum(axis=1)

        topic_content = topic[1]
        bm25_res = bm25_transform(corpus=[doc], topic=topic_content)
        bm25_feature = np.array([bm25_res]).transpose()

        test_features = np.array([[tf_feature[0, 0], idf_feature[0, 0], tfidf_feature[0, 0], bm25_feature[0, 0]]])

    res = model.predict_proba(test_features)

    return 0.0 if len(res[0]) == 1 else res[0][1]


# What do to in the absence of relevance feedback?
# Presence of relevance feedback: confusion matrix / accuracy / sensitivity / specificity ...
def evaluate(topics, d_test, r_test, classes_list):
    evaluation = []
    for i, topic in enumerate(topics):
        print(f"Evaluating topic {topic}")
        target = create_target(r_test[topic-101], d_test, max_values=len(classes_list[0]))
        classes = classes_list[i]

        # With the original continuous probability values (using regression metrics)
        cont_target = [float(x) for x in target]
        cont_classes = classes
        mean_squared_error = metrics.mean_squared_error(cont_target, cont_classes)
        mean_absolute_error = metrics.mean_absolute_error(cont_target, cont_classes)
        explained_variance_score = metrics.explained_variance_score(cont_target, cont_classes)
        r2_score = metrics.r2_score(cont_target, cont_classes)

        # With binarization of probability values (p >= bin_prob_threshold is considered relevant, else irrelevant)
        bin_target = target
        bin_classes = [1 if x >= bin_prob_threshold else 0 for x in classes]
        tp, tn, fp, fn = 0, 0, 0, 0
        for t, c in zip(bin_target, bin_classes):
            if t == 1 and c == 1:
                tp += 1
            elif t == 1 and c == 0:
                fn += 1
            elif t == 0 and c == 1:
                fp += 1
            elif t == 0 and c == 0:
                tn += 1
        accuracy = float(tp + tn) / (tp + fp + fn + tn)
        sens = float(tp) / (tp + fn) if tp + fn != 0 else 1.0
        spec = float(tn) / (tn + fp) if tn + fp != 0 else 1.0

        # Put everything in a tuple
        evaluation.append((mean_squared_error, mean_absolute_error, explained_variance_score, r2_score,
                           tp, fn, fp, tn, accuracy, sens, spec))

    return evaluation


def main():
    print("Processing d_train")
    train_corpus = process_documents(corpus_directory, train=True)  # Stemmed documents
    # train_corpus = process_documents(corpus_directory, train=True, stemmed=False)  # Non stemmed documents

    print("Processing d_test")
    test_corpus = process_documents(corpus_directory, train=False)  # Stemmed documents
    # test_corpus = process_documents(corpus_directory, train=False, stemmed=False)  # Non stemmed documents

    train_rels = extract_relevance(qrels_train_directory)
    test_rels = extract_relevance(qrels_test_directory)

    classes_list = []
    for topic_id in topic_ids:
        topic = process_topic(topic_id, topic_directory)
        print(f"Training for topic {topic_id}")
        model = training(topic, train_corpus, train_rels, model=KNeighborsClassifier(n_neighbors=50))
        # model = training(topic, train_corpus, train_rels, model=MultinomialNB(alpha=1.0))
        print(f"Classifying for topic {topic_id}")
        classes = [classify(test_corpus[i][1], topic, model) for i in range(len(test_corpus))]
        classes_list.append(classes)

    evaluation = evaluate(topic_ids, test_corpus, test_rels, classes_list)
    print()
    for i, topic_id in enumerate(topic_ids):
        print(f"***Statistics for topic {topic_id}***")
        print()
        mse, mae, evs, r2, tp, fn, fp, tn, acc, sens, spec = evaluation[i]
        print("**With the original continuous probability values (using regression metrics)**")
        print()
        print(f"Mean squared error: {mse}")
        print(f"Mean absolute error: {mae}")
        print(f"Explained variance score: {evs}")
        print(f"R^2 score: {r2}")
        print()
        print(f"**With binarization of probability values (p > "
              f"{bin_prob_threshold} is considered relevant, else irrelevant)**")
        print()
        print(f"True positives: {tp}")
        print(f"False negatives: {fn}")
        print(f"False positives: {fp}")
        print(f"True negatives: {tn}")
        print(f"Accuracy score: {acc}")
        print(f"Sensitivity: {sens}")
        print(f"Specificity: {spec}")
        print()
        print()


if __name__ == "__main__":
    main()
