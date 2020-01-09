from sklearn.feature_extraction.text import TfidfVectorizer

import vec_rep
from sklearn.feature_extraction.text import TfidfVectorizer

import vec_rep


def svm_classifier(model):
    tfidf_vec_tr = TfidfVectorizer(model)
    tfidf_doc_vec = tfidf_vec_tr.fit_transform(model.vocab)
    print("Ebola")


def cnn_classifier(model):
    pass


def rnn_classifier(model):
    pass


def log_regression(model):
    pass


svm_classifier(vec_rep.word_2_vec_vectorizer())
