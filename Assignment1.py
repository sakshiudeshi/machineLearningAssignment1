# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from sklearn.linear_model import perceptron
from sklearn.feature_extraction.text import CountVectorizer

import os

top_path = "/Users/sakshiudeshi/Documents/SUTD/Academics/Term 1 Sept - Dec/Machine Learning/Assignment 1/data/train"

#Acq files
acq_files = os.listdir(top_path + "/acq")

acq_train = []
acq_vectorizer = CountVectorizer(stop_words='english')


for f in acq_files:
	fd = open(top_path + "/acq/" + f, 'r')
	acq_train.append(fd.read())

acq_train = tuple(acq_train)

acq_vec_train = acq_vectorizer.fit_transform(acq_train)


#print acq_vec

#housing files
housing_files = os.listdir(top_path + "/housing")

housing_train = []
housing_vectorizer = CountVectorizer(stop_words='english')


for f in housing_files:
	fd = open(top_path + "/housing/" + f, 'r')
	housing_train.append(fd.read())

housing_train = tuple(housing_train)

housing_vec_train = housing_vectorizer.fit_transform(housing_train)

print housing_vec


