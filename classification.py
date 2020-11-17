from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from tfidf import *
# from bm25vectorizer import *

from scipy.sparse import hstack

bin_prob_threshold = 0.4  # Value between 0 and 1, probability values >= this threshold will be regarded as relevant
all_features = False  # Run classification with all features (TF, IDF, TF-IDF, BM25) or just TF-IDF

tf_vectorizer = CountVectorizer()
idf_vectorizer = TfidfVectorizer()
tfidf_vectorizer = TfidfVectorizer()
# bm25_vectorizer = BM25()


def create_target(rels, corpus, max_values=None):
    if max_values is None:
        max_values = len(corpus)
    else:
        max_values = min(max_values, len(corpus))
    target = []
    j = 0
    for i in range(max_values):
        if rels[j][0] == corpus[i][0]:
            target.append(rels[j][1])
            j += 1
        else:
            target.append(0)
    return target


def training(topic_id, d_train, r_train, model):
    topic_id -= 101
    rels = r_train[topic_id]

    global tf_vectorizer
    global idf_vectorizer
    global tfidf_vectorizer
    # global bm25_vectorizer

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_feature = tfidf_vectorizer.fit_transform(el[1] for el in d_train)
    features = tfidf_feature

    if all_features:
        tf_vectorizer = CountVectorizer()
        tf_feature = tf_vectorizer.fit_transform(el[1] for el in d_train)

        corpus_no_dups = remove_duplicates_corpus(d_train)
        idf_vectorizer = TfidfVectorizer()
        idf_feature = idf_vectorizer.fit_transform(el[1] for el in corpus_no_dups)

        """
        bm25_vectorizer = BM25()
        bm25_vectorizer.fit(X=(el[1] for el in d_train))
        bm25_feature = bm25_vectorizer.transform(X=(el[1] for el in d_train))
        """

        features = hstack([tf_feature, idf_feature, tfidf_feature])

    # Creates relevance list to be used to train the model
    train_target = create_target(rels, d_train)

    model.fit(X=features, y=train_target)

    return model


def classify(doc, model):
    tfidf_test_feature = tfidf_vectorizer.transform([doc])
    test_features = tfidf_test_feature

    if all_features:
        tf_test_feature = tf_vectorizer.transform([doc])
        no_dups_doc = remove_duplicates_doc(doc)
        idf_test_feature = idf_vectorizer.transform([no_dups_doc])
        # bm25_test_feature = bm25_vectorizer.transform(X=[doc])
        test_features = hstack([tf_test_feature, idf_test_feature, tfidf_test_feature])

    res = model.predict_proba(test_features)[0]
    return 0.0 if len(res) == 1 else res[1]


# What do to in the absence of relevance feedback?
# Presence of relevance feedback: confusion matrix / accuracy / sensitivity / specificity ...
def evaluate(topic_ids, d_test, r_test, classes_list):
    evaluation = []
    for i, topic in enumerate(topic_ids):
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

    topics = [101, 102, 103, 104, 105]
    classes_list = []
    for topic in topics:
        print(f"Training for topic {topic}")
        # model = training(topic, train_corpus, train_rels, model=KNeighborsClassifier(n_neighbors=50))
        model = training(topic, train_corpus, train_rels, model=MultinomialNB(alpha=1.0))
        print(f"Classifying for topic {topic}")
        classes = [classify(test_corpus[i][1], model) for i in range(docs_to_test)]
        classes_list.append(classes)

    evaluation = evaluate(topics, test_corpus, test_rels, classes_list)
    print()
    for i, topic in enumerate(topics):
        print(f"***Statistics for topic {topic}***")
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
