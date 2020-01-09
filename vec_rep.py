import multiprocessing
import os.path

import gensim
from bert_embedding import BertEmbedding
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_numeric, strip_multiple_whitespaces, remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import load_data_set

train_list, validation_list, test_list = load_data_set.load_moroco_data_set()
train_samples = train_list[1]
train_samples = [x.replace("$NE$", '') for x in train_samples]


def read_stop_words():
    stop_words = []
    with open(r"C:\Users\thomi\PycharmProjects\ProiectIA\romanian_stop_words.txt", "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            stop_words.append(line.replace('\n', ''))
    return list(set(stop_words))


# TF–IDF, short for term frequency–inverse document frequency, is a numerical
# statistic that is intended to reflect how important a word is to a document in a collection or corpus.
# It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.
# The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by
# the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words
# appear more frequently in general.
def tf_idf_vectorizer():
    print("TF-IDF:")
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, encoding='utf-8')
    tfidf_vectorizer.stop_words = read_stop_words()
    tfidf_vectorizer.fit_transform(train_samples)
    print(tfidf_vectorizer.get_feature_names())
    return tfidf_vectorizer


# BOW, short for bag of words, marks the number of appearances of each word in a text
def bow_vectorizer():
    global train_samples
    print("Bag Of Words:")
    bow_vect = CountVectorizer(encoding='utf-8')
    bow_vect.stop_words = read_stop_words()
    bow_vect.fit_transform(train_samples)
    print(bow_vect.get_feature_names())
    return bow_vect


def word_2_vec_vectorizer():
    model = None
    if os.path.exists('model.bin'):
        print("Model exists!")
        model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', fvocab="vocab.txt", binary=True)
        return model

    gensim.parsing.preprocessing.STOPWORDS = read_stop_words()
    print("Stripping all the stop words...")
    prep_train_samples = [remove_stopwords(x) for x in train_samples]
    print("Stripped all the stop words!")

    print("Stripping all the numbers...")
    prep_train_samples = [strip_numeric(x) for x in train_samples]
    print("Stripped all the numbers!")

    print("Stripping all the dangling whitespaces...")
    prep_train_samples = [strip_multiple_whitespaces(x) for x in train_samples]
    print("Stripped all the dangling whitespaces!")

    print("Tokenizing...")
    prep_train_samples = [gensim.utils.simple_preprocess(x) for x in train_samples]
    print("Tokenization done!")

    workers = multiprocessing.cpu_count()
    print('number of cpu: {}'.format(workers))
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise."

    print("Creating model...")
    model = gensim.models.Word2Vec(prep_train_samples, size=150, window=10, min_count=2, workers=workers, iter=10)
    print("Model created!")

    model.wv.save_word2vec_format('model.bin', fvocab="vocab.txt", binary=True)
    print("Model saved!")

    return model


def bert_vectorizer():
    bert = BertEmbedding()
    result = bert(train_samples[:10])
    print(result)


def fast_text_vectorizer():
    pass


# tf_idf_vectorizer()
# bow_vectorizer()
# word_2_vec_vectorizer()
# bert_vectorizer()
