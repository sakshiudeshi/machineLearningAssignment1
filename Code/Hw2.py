#Code for Avergaged Perceptron
from __future__ import division
import numpy as np
import os

train_path = os.getcwd() + "/data/train/"
test_path = os.getcwd() + "/data/test/"

HOUSING_CONS = 1
ACQ_CONS = -1

file_types = []
all_train_texts = dict()

#building vocabulary
def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
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


def sign(vector):
    res = np.dot(vector, theta) + theta_o;
    if res >= 0:
        return HOUSING_CONS
    else:
        return ACQ_CONS

def get_files(path):
    all_texts = dict()
    for file_type in sorted(os.listdir(path)):
        if not file_type.startswith('.'):
            file_types.append(file_type)
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

def test_texts(type, CONS, all_test_texts):
    doc_term_matrix_test = vectorize(all_test_texts[type])
    hits = 0
    for item in doc_term_matrix_test:
        if(sign(item) == CONS):
            hits = hits + 1
            print "HIT"
        else:
            print "MISS"
    return hits


vocabulary = build_lexicon(get_corpus())

all_train_texts = get_files(train_path)

#transforming training data to vectors
acq_texts = all_train_texts["acq"]
acq_doc_term_matrix_train = vectorize(acq_texts)


housing_texts = all_train_texts["housing"]
housing_doc_term_matrix_train = vectorize(housing_texts)


#train using data
train_dict = dict()
train_dict[HOUSING_CONS] = housing_doc_term_matrix_train
train_dict[ACQ_CONS] = acq_doc_term_matrix_train

#initializing thetha vector
theta_o = 0;
theta = []
for i in range(0, len(vocabulary)):
    theta.append(0)

thetas = []
theta_os = []
thetas.append(theta)
theta_os.append(theta_o)


while True:
    acq_trained = True
    housing_trained = True
    for item in acq_doc_term_matrix_train:
        if sign(item) != ACQ_CONS:
            item = [x * ACQ_CONS for x in item]
            theta = np.add(theta, item)
            theta_o = theta_o + ACQ_CONS
            thetas.append(theta)
            theta_os.append(theta_o)
            acq_trained = False


    for item in housing_doc_term_matrix_train:
        if sign(item) != HOUSING_CONS:
            item = [x * HOUSING_CONS for x in item]
            theta = np.add(theta, item)
            theta_o = theta_o + HOUSING_CONS
            thetas.append(theta)
            theta_os.append(theta_o)
            housing_trained = False

    if acq_trained and housing_trained:
        break


#averaging theta and theta_o
theta = np.mean(thetas, axis = 0)
theta_o = np.mean(theta_os)


all_test_texts = get_files(test_path)
acq_hits = test_texts('acq',ACQ_CONS, all_test_texts)
housing_hits = test_texts('housing', HOUSING_CONS, all_test_texts)
acq_tests = len(all_test_texts['acq'])
housing_tests = len(all_test_texts['housing'])


print "Accuracy for ACQ " + str((acq_hits/acq_tests) * 100) + "%"
print "Accuracy for Housing " + str((housing_hits/housing_tests) * 100) + "%"
print "Total accuracy " + str(((acq_hits+housing_hits)/(acq_tests+ housing_tests)) * 100) + "%"