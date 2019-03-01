#!/bin/python 

import numpy as np
from random import shuffle
import os
from sklearn.svm.classes import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import cPickle
import sys
from imblearn.over_sampling import SMOTE, ADASYN

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    print event_name,feat_dir,feat_dim,output_file
    X_train = []
    Y_train = []
    # svm_classifier = SVC(probability=True,kernel='rbf',C=1,class_weight={1:0.6},gamma='scale')
    # svm_classifier = SVC(gamma='scale',kernel="rbf")
    # svm_classifier = LogisticRegression(penalty='l1')

    #Try to define a max depth????
    # Try 450 cluster size????
    svm_classifier = RandomForestClassifier(n_estimators=300)


    # svm_classifier = GradientBoostingClassifier()
    train_files_list = open('list/train')
    train_files = train_files_list.readlines()
    val_files_list = open('list/val')
    merged_file_list = open('merged_train')
    # train_files = train_files_list.readlines() + val_files_list.readlines()
    # shuffle(merged_file_list)
    for line in train_files:
        file,tag = line.strip().split()
        f_file = open(sys.argv[2]+file+'.txt')
        X_train.append(np.array(map(float,f_file.read().strip().split())))
        f_file.close()
        if tag==event_name:            
            Y_train.append(1)
        else:
            Y_train.append(0)
    X_train =  np.array(X_train)
    print X_train.shape
    # X_resampled, Y_resampled = SMOTE().fit_resample(X_train, Y_train)
    svm_classifier.fit(X_train,Y_train)
    with open(output_file+'.pkl', 'wb') as fid:
        cPickle.dump(svm_classifier, fid)



    # event_name = sys.argv[1]
    # feat_dir = sys.argv[2]
    # feat_dim = int(sys.argv[3])
    # output_file = 'multi_svm'
    # X_train = []
    # Y_train = []
    # svm_classifier = SVC(probability=True)
    # # svm_classifier = LogisticRegression(class_weight={0:1,1:50,2:50,3:50})
    # # svm_classifier = RandomForestClassifier()
    # # svm_classifier = GradientBoostingClassifier(subsample=0.1)
    # train_files_list = open('list/train')
    # for line in train_files_list.readlines():
    #     file,tag = line.strip().split()
    #     f_file = open(sys.argv[2]+file+'.txt')
    #     X_train.append(map(float,f_file.read().strip().split()))
    #     f_file.close()
    #     if tag=="P001":
    #         Y_train.append(1)
    #     elif tag=="P002":
    #         Y_train.append(2)
    #     elif tag=="P003":
    #         Y_train.append(3)
    #     else:
    #         Y_train.append(0)
    # X_train =  np.array(X_train)
    # Y_train = np.array(Y_train)
    # # X_resampled, Y_resampled = SMOTE().fit_resample(X_train, Y_train)
    # svm_classifier.fit(X_train,Y_train)
    # with open(output_file+'.pkl', 'wb') as fid:
    #     cPickle.dump(svm_classifier, fid)




    print 'SVM trained successfully for event %s!' % (event_name)
