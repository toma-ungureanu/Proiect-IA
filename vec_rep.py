import multiprocessing
import os.path

import gensim
import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.parsing.preprocessing import strip_numeric, strip_multiple_whitespaces, remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import load_data_set

workers = multiprocessing.cpu_count()
print('Number of cpu(s): {}'.format(workers))
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise."


def preprocess_data():
    train_list, validation_list, test_list = load_data_set.load_moroco_data_set()
    train_list[1] = [x.replace("$NE$", '') for x in train_list[1]]
    validation_list[1] = [x.replace("$NE$", '') for x in validation_list[1]]
    test_list[1] = [x.replace("$NE$", '') for x in test_list[1]]
    return train_list, validation_list, test_list


def csv_data_frame():
    header = ['id', 'info', 'category']
    train_list, validation_list, test_list = preprocess_data()
    train_data_frame = pd.DataFrame(zip(train_list[0], train_list[1], train_list[3]), columns=header)
    test_data_frame = pd.DataFrame(zip(test_list[0], test_list[1], test_list[3]), columns=header)
    validation_data_frame = pd.DataFrame(zip(validation_list[0], validation_list[1], validation_list[3]),
                                         columns=header)
    return train_data_frame, validation_data_frame, test_data_frame


def read_stop_words():
    print("Reading the stop-words...")
    dirname = os.path.dirname(__file__)
    stop_words = []
    with open(os.path.join(dirname, "romanian_stop_words.txt"), "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            stop_words.append(line.replace('\n', ''))
    print("Read the stop words!")
    return list(set(stop_words))


def clean_samples(samples):
    gensim.parsing.preprocessing.STOPWORDS = read_stop_words()

    print("Stripping all the stop words...")
    prep_train_samples = [remove_stopwords(x.lower()) for x in samples]
    print("Stripped all the stop words!")

    print("Stripping all the numbers...")
    prep_train_samples = [strip_numeric(x) for x in prep_train_samples]
    print("Stripped all the numbers!")

    print("Stripping all the dangling whitespaces...")
    prep_train_samples = [strip_multiple_whitespaces(x) for x in prep_train_samples]
    print("Stripped all the dangling whitespaces!")

    print("Tokenizing...")
    prep_train_samples = [gensim.utils.simple_preprocess(x) for x in prep_train_samples]
    print("Tokenization done!")
    return prep_train_samples


def tokenize_data():
    if os.path.exists(r'csv\train.csv') and os.path.exists(r'csv\validate.csv') and os.path.exists(r'csv\test.csv'):
        print("Data frame exists!")
        train = pd.read_csv(r'csv\train.csv')
        test = pd.read_csv(r'csv\test.csv')
        validate = pd.read_csv(r'csv\validate.csv')
        return train, test, validate

    print("Creating data frame...")
    train, validate, test = csv_data_frame()
    train['info'] = clean_samples(train['info'])
    validate['info'] = clean_samples(validate['info'])
    test['info'] = clean_samples(test['info'])

    header = ['id', 'info', 'category']
    train.to_csv(r'csv\train.csv', header=header, index=None)
    test.to_csv(r'csv\test.csv.', header=header, index=None)
    validate.to_csv(r'csv\validate.csv', header=header, index=None)
    print("Done!")
    return train, validate, test


# TF–IDF, short for term frequency–inverse document frequency, is a numerical
# statistic that is intended to reflect how important a word is to a document in a collection or corpus.
# It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.
# The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by
# the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words
# appear more frequently in general.
def tf_idf_vectorizer(data):
    print("TF-IDF:")
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, encoding='utf-8', min_df=30)
    tfidf_vectorizer.stop_words = read_stop_words()
    return tfidf_vectorizer.fit_transform(data).toarray()


# BOW, short for bag of words, marks the number of appearances of each word in a text
def bow_vectorizer(data):
    print("Bag Of Words:")
    bow_vect = CountVectorizer(encoding='utf-8')
    bow_vect.stop_words = read_stop_words()
    bow_vect.fit_transform(data)
    return bow_vect


def word_2_vec_vectorizer(data):
    global workers
    model = None
    if os.path.exists(r'word2vec\model.bin'):
        print("Model exists!")
        model = gensim.models.KeyedVectors.load_word2vec_format(r'word2vec\model.bin', fvocab=r"word2vec\vocab.txt",
                                                                binary=True)
        return model

    print("Creating model...")
    model = gensim.models.Word2Vec(data, size=150, window=10, min_count=2, workers=workers, iter=10)
    print("Model created!")

    print("Saving model...")
    model.wv.save_word2vec_format(r'word2vec\model.bin', fvocab=r'word2vec\vocab.txt', binary=True)
    print("Model saved!")
    return model


def bert_vectorizer():
    pass


def fast_text_vectorizer(data):
    model = None
    if os.path.exists(r'fast_text\model.bin'):
        print("Model exists!")
        model = gensim.models.FastText.load(fname_or_handle=r'fast_text\model.bin')
        return model

    print("Creating model...")
    model = FastText(data, size=100, window=5, min_count=5, workers=4, sg=1)
    print("Model created!")

    print("Saving model...")
    model.save(fname_or_handle=r'fast_text\model.bin')
    print("Model saved!")
    return model


def main():
    # tf_idf_vectorizer(tokenize_data()[0]['info'])
    # bow_vectorizer(tokenize_data()[0]['info'])
    # word_2_vec_vectorizer(tokenize_data()[0]['info'])
    fast_text_vectorizer(tokenize_data()[0]['info'])
    # bert_vectorizer()
