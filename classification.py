from core import *

from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from scipy.spatial.distance import cosine
from math import isnan
import numpy as np

#######################################################################################################################

# Customize parameters here:

classifier = KNeighborsClassifier(n_neighbors=25, metric="euclidean")
# classifier = MultinomialNB()
# classifier = MLPClassifier((100, 100, 100), max_iter=400)

all_features = False       # Run classification with all features (TF, IDF, TF-IDF, BM25) or just TF-IDF
bin_prob_threshold = 0.3   # Value between 0 and 1, probability values >= this threshold will be regarded as relevant

#######################################################################################################################


np.set_printoptions(threshold=10)
np.seterr(invalid="ignore")


tf_vectorizer = CountVectorizer()
idf_vectorizer = TfidfVectorizer()
tfidf_vectorizer = TfidfVectorizer()
bm25_vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)


# Creates target vector based on relevance feedback
def create_target(rels, corpus, max_values=None):
    if max_values is None:
        max_values = len(corpus)
    else:
        max_values = min(max_values, len(corpus))
    target = []
    j = 0
    for i in range(max_values):
        if j == len(rels) or rels[j][0] != corpus[i][0]:
            target.append(0)
        else:
            target.append(rels[j][1])
            j += 1

    return target


avdl = 0


def training(topic, d_train, r_train, model):
    topic_id = topic[0] - 101
    rels = r_train[topic_id]

    global tf_vectorizer
    global idf_vectorizer
    global tfidf_vectorizer
    global bm25_vectorizer

    # Uses TF-IDF vectorizer to calculate TF-IDF feature (cosine distance)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(el[1] for el in d_train)
    doc_tfidf_vec = tfidf_vectorizer.transform(el[1] for el in d_train)
    topic_tfidf_vec = tfidf_vectorizer.transform([topic[1]])
    tfidf_cosine_distances = [cosine(doc_tfidf_vec[i].toarray(), topic_tfidf_vec[0].toarray())
                              for i in range(doc_tfidf_vec.shape[0])]
    # If distance is NaN, consider maximum distance (1)
    tfidf_feature = np.array([[1 if isnan(tfidf_cosine_distances[i]) else tfidf_cosine_distances[i]
                               for i in range(doc_tfidf_vec.shape[0])]]).transpose()

    features = tfidf_feature

    if all_features:
        # Uses count vectorizer to calculate TF feature (sum of TFs)
        tf_vectorizer = CountVectorizer()
        tf_vectorizer.fit([topic[1]])
        tf_doc = tf_vectorizer.transform(el[1] for el in d_train)
        tf_feature = tf_doc.sum(axis=1)

        # Uses TF-IDF vectorizer to calculate IDF feature (cosine distance of TF-IDF vectors with all TFs at 1)
        idf_vectorizer = TfidfVectorizer()
        corpus_no_dups = remove_duplicates_corpus(d_train)
        idf_vectorizer.fit(el[1] for el in corpus_no_dups)
        topic_no_dups = remove_duplicates_doc(topic[1])
        topic_idf_vec = idf_vectorizer.transform([topic_no_dups])
        doc_idf_vec = idf_vectorizer.transform(el[1] for el in d_train)
        idf_cosine_distances = [cosine(doc_idf_vec[i].toarray(), topic_idf_vec[0].toarray())
                                for i in range(doc_idf_vec.shape[0])]
        idf_feature = np.array([[1 if isnan(idf_cosine_distances[i]) else idf_cosine_distances[i]
                                 for i in range(doc_idf_vec.shape[0])]]).transpose()

        # Uses TF-IDF vectorizer to calculate BM25 feature (BM25 score)
        bm25_vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        corpus = [el[1] for el in d_train]
        bm25_fit(corpus)
        bm25_res = bm25_transform(corpus, topic[1])
        bm25_feature = np.array([[el for el in bm25_res]]).transpose()

        # Creates an array of 4-element arrays, where each element corresponds to a different feature
        features = hstack([tf_feature, idf_feature, tfidf_feature, sparse.csc_matrix(np.array(bm25_feature))]).toarray()

    # Creates relevance list to be used to train the model
    train_target = create_target(rels, d_train)

    model.fit(X=features, y=train_target)

    return model


# Learns vocabulary, idf and avdl for BM25 score computation
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


# Computes BM25 score of the topic against the corpus
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
    # All features are computed in a similar fashion to what is done in the training function

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
        topic_no_dups = remove_duplicates_doc(doc)
        idf_topic = idf_vectorizer.transform([topic_no_dups])
        cosine_distance = cosine(idf_doc[0].toarray(), idf_topic[0].toarray())
        idf_feature = np.array([[1 if isnan(cosine_distance) else cosine_distance]])

        topic_content = topic[1]
        bm25_res = bm25_transform(corpus=[doc], topic=topic_content)
        bm25_feature = np.array([bm25_res]).transpose()

        test_features = np.array([[tf_feature[0, 0], idf_feature[0, 0], tfidf_feature[0, 0], bm25_feature[0, 0]]])

    res = model.predict_proba(test_features)

    # If len(res[0]) == 1, that means there is only one class, which means the
    # classifier for this topic was only trained with documents that have negative feedback
    return 0.0 if len(res[0]) == 1 else res[0][1]


def evaluate(topics, d_test, r_test, classes_list):
    evaluation = []
    for i, topic in enumerate(topics):
        print(f"Evaluating topic {topic}")
        target = create_target(r_test[topic-101], d_test, max_values=len(classes_list[0]))
        classes = classes_list[i]

        # With the original continuous probability values (using regression metrics)
        cont_target = [float(x) for x in target]
        cont_classes = classes
        # Mean Squared Error regression loss
        mse = mean_squared_error(cont_target, cont_classes)
        # Mean Absolute Error regression loss
        mae = mean_absolute_error(cont_target, cont_classes)
        # Explained Variance regression score function, measures how well it takes variation into account in the corpus
        evs = explained_variance_score(cont_target, cont_classes)
        # R^2 correlation measure
        r2 = r2_score(cont_target, cont_classes)

        # With binarization of probability values (p >= bin_prob_threshold is considered relevant, else irrelevant)
        bin_target = target
        bin_classes = [1 if x >= bin_prob_threshold else 0 for x in classes]
        # Confusion matrix
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
        # Accuracy, sensitivity and specificity
        accuracy = float(tp + tn) / (tp + fp + fn + tn)
        sens = float(tp) / (tp + fn) if tp + fn != 0 else 1.0
        spec = float(tn) / (tn + fp) if tn + fp != 0 else 1.0

        # Puts everything in a tuple
        evaluation.append((mse, mae, evs, r2, tp, fn, fp, tn, accuracy, sens, spec))

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
        model = training(topic, train_corpus, train_rels, model=classifier)
        # model = training(topic, train_corpus, train_rels, model=MultinomialNB(alpha=1.0))
        print(f"Classifying for topic {topic_id}")
        classes = [classify(test_corpus[i][1], topic, model) for i in range(len(test_corpus))]
        classes_list.append(classes)

    evaluation = evaluate(topic_ids, test_corpus, test_rels, classes_list)
    print()
    list_of_r2 = []
    for i, topic_id in enumerate(topic_ids):
        print(f"***Statistics for topic {topic_id}***")
        # print()
        mse, mae, evs, r2, tp, fn, fp, tn, acc, sens, spec = evaluation[i]
        print("**With the original continuous probability values (using regression metrics)**")
        print()
        print(f"Mean squared error: {mse}")
        print(f"Mean absolute error: {mae}")
        print(f"Explained variance score: {evs}")
        print(f"R^2 score: {r2}")
        list_of_r2.append(r2)
        print()
        print(f"**With binarization of probability values (p >= "
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
