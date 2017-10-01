from sklearn.linear_model import perceptron
from sklearn.feature_extraction.text import CountVectorizer

import os

import numpy as np

train_top_path = "/Users/sakshiudeshi/Documents/SUTD/Academics/Term 1 Sept - Dec/Machine Learning/Assignment 1/data/train"

#Acq files
acq_files = os.listdir(train_top_path + "/acq")

acq_train = []
acq_vectorizer = CountVectorizer(stop_words='english')


for f in acq_files:
	fd = open(train_top_path + "/acq/" + f, 'r')
	acq_train.append(fd.read())

acq_train = tuple(acq_train)

acq_vec_train = acq_vectorizer.fit_transform(acq_train)



#housing files
housing_files = os.listdir(train_top_path + "/housing")

housing_train = []
housing_vectorizer = CountVectorizer(stop_words='english')


for f in housing_files:
	fd = open(train_top_path + "/housing/" + f, 'r')
	housing_train.append(fd.read())

housing_train = tuple(housing_train)

housing_vec_train = housing_vectorizer.fit_transform(housing_train)

print type(housing_vec_train)


def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 20

    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + eta*X[i]*Y[i]

    return w

print perceptron_sgd(acq_vec_train.toarray(), housing_vec_train.toarray())
