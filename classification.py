import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB

import vec_rep

tf_dir = r'tf-idf_arrays'
train_path = os.path.join(tf_dir, 'train.data')
test_path = os.path.join(tf_dir, 'test.data')
validate_path = os.path.join(tf_dir, 'vaidation.data')
naive_bayes_filename = r'fitted_models\naive_bayes.sav'


def exists(func):
    data = vec_rep.tokenize_data()
    train_x = None
    test_x = None
    validate_x = None
    if os.path.exists(train_path) and os.path.getsize(train_path) > 0 and \
            os.path.exists(test_path) and os.path.getsize(test_path) > 0 and \
            os.path.exists(validate_path) and os.path.getsize(validate_path) > 0:
        with open(train_path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            train_x = unpickler.load()
        with open(test_path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            test_x = unpickler.load()
        with open(train_path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            validate_x = unpickler.load()
        return train_x, validate_x, test_x

    train_x = func(data[0]['info'])
    pickle.dump(train_x, open(train_path, 'wb'))

    validate_x = func(data[1]['info'])
    pickle.dump(validate_x, open(validate_path, 'wb'))

    test_x = func(data[2]['info'])
    pickle.dump(test_x, open(test_path, 'wb'))

    return train_x, validate_x, test_x


def naive_bayes(func):
    global naive_bayes_filename

    train_x, validate_x, test_x = exists(func)
    data = vec_rep.tokenize_data()
    train_y = data[0]['category']
    validate_y = data[1]['category']
    test_y = data[2]['category']

    model = None

    if os.path.exists(naive_bayes_filename) and os.path.getsize(naive_bayes_filename) > 0:
        print("Model exists!")
        with open(naive_bayes_filename, "rb") as f:
            unpickler = pickle.Unpickler(f)
            model = unpickler.load()

    else:
        mnb = MultinomialNB()
        print("Training hard...")
        model = mnb.fit(train_x, train_y)
        print("Huh, I trained for a thousand years, let's do this!")

        print("Saving model...")
        pickle.dump(model, open(naive_bayes_filename, 'wb'))
        print("Model saved!")

    model.predict(test_x)


def svm_classifier(model):
    tfidf_vec_tr = TfidfVectorizer(model)
    tfidf_doc_vec = tfidf_vec_tr.fit_transform(model.vocab)


def cnn_classifier(model):
    pass


def rnn_classifier(model):
    pass


def log_regression(model):
    pass


naive_bayes(vec_rep.tf_idf_vectorizer)
