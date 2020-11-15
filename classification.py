from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from tfidf import *


vectorizer = TfidfVectorizer()


def training(topic_id, d_train, r_train, classifier=MultinomialNB()):
    topic_id -= 101
    rels = r_train[topic_id]

    global vectorizer
    vectorizer, matrix = vectorize_corpus(d_train)

    # Creates relevance list to be used to train the model
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


def classify(doc, model):
    test_vec = vectorizer.transform([doc])
    return model.predict_proba(test_vec)[0][1]


def evaluate():
    pass


def main():
    train_corpus = process_documents(corpus_directory, train=True)  # Stemmed documents
    # train_corpus = process_documents(corpus_directory, train=True, stemmed=False)  # Non stemmed documents

    classifiers = [MultinomialNB(), KNeighborsClassifier()]

    rels = extract_relevance(qrels_train_directory)
    classifier = training(102, train_corpus, rels, classifier=classifiers[1])

    test_corpus = process_documents(corpus_directory, train=False)  # Stemmed documents
    # test_corpus = process_documents(corpus_directory, train=False, stemmed=False)  # Non stemmed documents

    probs = [(test_corpus[i][0], classify(test_corpus[i][1], classifier)) for i in range(1000)]
    print(probs)


if __name__ == "__main__":
    main()
