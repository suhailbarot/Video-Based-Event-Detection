import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import cPickle
import sys
import csv



f1 = open('cnn_pred/kaggle_test_kmeans_cnnP001.csv')
f2 = open('cnn_pred/kaggle_test_kmeans_cnnP002.csv')
f3 = open('cnn_pred/kaggle_test_kmeans_cnnP003.csv')


r1 = [x for x in csv.reader(f1)]
r2 = [x for x in csv.reader(f2)]
r3 = [x for x in csv.reader(f3)]

sr1 = sum([float(x[1]) for x in r1])
nr1 = [(x[0],float(x[1])/float(sr1)) for x in r1]

sr2 = sum([float(x[1]) for x in r2])
nr2 = [(x[0],float(x[1])/float(sr2)) for x in r2]

sr3 = sum([float(x[1]) for x in r3])
nr3 = [(x[0],float(x[1])/float(sr3)) for x in r3]

output_file = open('kaggle_test_kmeans_final.csv','w+')
output_file.write("VideoID,label\n")

for i,j,k in zip(nr1,nr2,nr3):
	tag = np.argmax(np.array([i[1],j[1],k[1]]))+1
	output_file.write(i[0]+","+str(tag)+"\n")







