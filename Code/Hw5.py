#code for Sub Gradient Descent with Regularization
from __future__ import division
import numpy as np
import os
import random

train_path = os.getcwd() + "/data/train/"
test_path = os.getcwd() + "/data/test/"

gamma = 0.001

HOUSING_CONS = 1
ACQ_CONS = -1

learning_rate = 0.001

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


def result(vector):
    result = np.dot(vector, theta) + theta_o;
    return result

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

    print type + " - Hits : " + str(hits) + " Misses : " + str(len(doc_term_matrix_test) - hits)
    return hits


vocabulary = build_lexicon(get_corpus())

all_train_texts = get_files(train_path)

#transforming training data to vectors
acq_texts = all_train_texts["acq"]
acq_doc_term_matrix_train = vectorize(acq_texts)


housing_texts = all_train_texts["housing"]
housing_doc_term_matrix_train = vectorize(housing_texts)

all_doc_term_matrix_train = []
for item in acq_doc_term_matrix_train:
    all_doc_term_matrix_train.append(item)

for item in housing_doc_term_matrix_train:
    all_doc_term_matrix_train.append(item)

print len(all_doc_term_matrix_train)



#initializing thetha vector
theta_o = 0;
theta = []
for i in range(0, len(vocabulary)):
    theta.append(0)



def calc_loss():
    sum = 0;
    n = len(all_doc_term_matrix_train)
    for X in all_doc_term_matrix_train:
        if X in acq_doc_term_matrix_train:
            Y = ACQ_CONS
        else:
            Y = HOUSING_CONS
        output = result(X)
        cons = gamma*np.dot(theta, theta)
        sum = sum + max((1 - output*Y), 0)
    loss = sum/n + cons
    return loss


result_tracker = []


min_loss = 1000020201
min_theta = []
min_theta_o = 0
for x in range (1, 3000):
    X = random.choice(all_doc_term_matrix_train)
    if X in acq_doc_term_matrix_train:
        Y = ACQ_CONS
    else:
        Y = HOUSING_CONS
    output = result(X)
    if output*Y <= 1:
        item = [x*Y*learning_rate for x in X]
        new_theta = np.add(theta, item)
        theta = new_theta
    loss = calc_loss()
    print "Loss is " + str(loss)
    if(loss < min_loss):
        min_loss = loss
        min_theta = theta
        min_theta_o = theta_o

theta = min_theta
theta_o = min_theta_o

all_test_texts = get_files(test_path)

print ""
print "In Train Set"
acq_hits = test_texts('acq',ACQ_CONS, all_train_texts)
housing_hits = test_texts('housing', HOUSING_CONS, all_train_texts)
acq_tests = len(all_train_texts['acq'])
housing_tests = len(all_train_texts['housing'])

print "Gamma is " + str(gamma)
print "Accuracy for ACQ " + str((acq_hits/acq_tests) * 100) + "%"
print "Accuracy for Housing " + str((housing_hits/housing_tests) * 100) + "%"
tot_acc = ((acq_hits+housing_hits)/(acq_tests+ housing_tests)) * 100
print "Total accuracy " + str(tot_acc) + "%"

print ""
print "In Test Set"
acq_hits = test_texts('acq',ACQ_CONS, all_test_texts)
housing_hits = test_texts('housing', HOUSING_CONS, all_test_texts)
acq_tests = len(all_test_texts['acq'])
housing_tests = len(all_test_texts['housing'])

print "Gamma is " + str(gamma)
print "Accuracy for ACQ " + str((acq_hits/acq_tests) * 100) + "%"
print "Accuracy for Housing " + str((housing_hits/housing_tests) * 100) + "%"
tot_acc = ((acq_hits+housing_hits)/(acq_tests+ housing_tests)) * 100
print "Total accuracy " + str(tot_acc) + "%"