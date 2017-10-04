#Code for Multiclass Classification
from __future__ import division
import numpy as np
import os
import collections
import random

TYPE_CONS = 1
NON_TYPE_CONS = -1
learning_rate = 0.05

train_path = os.getcwd() + "/data/train/"
test_path = os.getcwd() + "/data/test/"
file_types = []

Params = collections.namedtuple('Params', ['theta', 'theta_o'])

#building vocabulary
def build_lexicon():
    lexicon = set()
    for doc in get_corpus():
        lexicon.update([word for word in doc.split()])
    return lexicon

def tf(term, document):
    return freq(term, document)


def freq(term, document):
    return document.split().count(term)


def get_corpus():
    corpus = []
    for file_type in sorted(os.listdir(train_path)):
        if not file_type.startswith('.'):
            file_types.append(file_type)
            for file_name in sorted(os.listdir(train_path + file_type)):
                fd = open(train_path + file_type + "/" + file_name, 'r')
                text = fd.read()
                corpus.append(text)

    return corpus


def get_files(path):
    all_texts = dict()
    for file_type in sorted(os.listdir(path)):
        if not file_type.startswith('.'):
            texts = []
            for file_name in sorted(os.listdir(path + file_type)):
                fd = open(path + file_type + "/" + file_name, 'r')
                text = fd.read()
                texts.append(text)
            all_texts[file_type] = texts
    return all_texts

def vectorize(texts):
    doc_term_matrix = []
    for doc in texts:
        vector = [tf(word, doc) for word in vocabulary]
        doc_term_matrix.append(vector)
    return doc_term_matrix

def result(vector, theta, theta_o):
    result = np.dot(vector, theta) + theta_o;
    return result

def sign(vector, theta, theta_o):
    if (result(vector, theta, theta_o) >= 0):
        return TYPE_CONS
    else:
        return NON_TYPE_CONS

def test_texts(type, CONS, all_test_texts, theta, theta_o):
    doc_term_matrix_test = vectorize(all_test_texts[type])
    hits = 0
    for item in doc_term_matrix_test:
        if(sign(item, theta, theta_o) == CONS):
            hits = hits + 1

    print type + " - Hits : " + str(hits) + " Misses : " + str(len(doc_term_matrix_test) - hits)
    return hits

def calc_loss(all_doc_term_matrix_train, type_doc_matrix_train, theta, theta_o):
    sum = 0;
    for X in all_doc_term_matrix_train:
        if X in type_doc_matrix_train:
            Y = TYPE_CONS
        else:
            Y = NON_TYPE_CONS
        output = result(X, theta, theta_o)
        sum = sum + max((1 - output*Y), 0)
    return sum

def get_classifier (type, all_train_texts):
    type_texts = all_train_texts[type]
    non_type_texts = []

    for item in all_train_texts:
        if type_texts != all_train_texts[item]:
            for article in all_train_texts[item]:
                non_type_texts.append(article)

    type_doc_matrix_train = vectorize(type_texts)
    non_type_doc_matrix_train = vectorize(non_type_texts)

    all_doc_term_matrix_train = []
    for item in type_doc_matrix_train:
        all_doc_term_matrix_train.append(item)

    for item in non_type_doc_matrix_train:
        all_doc_term_matrix_train.append(item)

    # initializing thetha vector
    theta_o = 0;
    theta = []
    for i in range(0, len(vocabulary)):
        theta.append(0)

    while True:
        X = random.choice(all_doc_term_matrix_train)
        if X in type_doc_matrix_train:
            Y = TYPE_CONS
        else:
            Y = NON_TYPE_CONS
        output = result(X, theta, theta_o = 0)
        if output * Y <= 1:
            item = [x * Y * learning_rate for x in X]
            new_theta = np.add(theta, item)
            theta = new_theta

        if (calc_loss(all_doc_term_matrix_train, type_doc_matrix_train, theta, theta_o) == 0):
            break

    p = Params(theta, theta_o)
    return p


vocabulary = build_lexicon()
all_text_test = get_files(test_path)
all_text_train = get_files(train_path)


for item in file_types:
    params = get_classifier(item, all_text_train)
    print "In Train Set"
    hits = test_texts(item, TYPE_CONS, all_text_train, params.theta, params.theta_o)
    tests = len(all_text_train[item])
    print "Accuracy for " + item + " " + str((hits / tests) * 100) + "%"
    print "In Test Set"
    hits = test_texts(item, TYPE_CONS, all_text_test, params.theta, params.theta_o)
    tests = len(all_text_test[item])
    print "Accuracy for " + item + " " + str((hits/tests) * 100) + "%"