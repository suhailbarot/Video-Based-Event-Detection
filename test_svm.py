#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    # model_file = 'multi_svm'
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = open(sys.argv[4],'w+')

    # # output_file = open('mfcc_pred/.lst','w+')

    svm_classifier = cPickle.load(open(model_file+".pkl"))
    X_val = []
    val_files_list = open('list/val')
    for line in val_files_list.readlines():
        file,_ = line.strip().split()
        f_file = open(sys.argv[2]+file+'.txt')
        X_val.append(np.array(map(float,f_file.read().strip().split())))
        f_file.close()
        # if tag=="NULL":
        #     Y_train.append(0)
        # else:
        #     Y_train.append(1)
    Y_val = svm_classifier.predict_proba(X_val)
    val_files_list.seek(0)
    for line,num_tag in zip(val_files_list.readlines(),Y_val):
        file,_ = line.strip().split()
        # if num_tag==1:
        #     output_file.write("1\n")
        # else:
        #     output_file.write("0\n")
        output_file.write(str(num_tag)+"\n")
    output_file.close()


    output_file = open('surf_pred/surf_test_'+sys.argv[5]+'.txt','w+')
    print "AFGFSASFG"
    # output_file = open('kaggle_test_kmeans.csv','w+')
    X_test = []
    test_files_list = open('list/all_test.video')
    for line in test_files_list.readlines():
        file = line.strip()
        f_file = open(sys.argv[2]+file+'.txt')
        X_test.append(np.array(map(float,f_file.read().strip().split())))
        f_file.close()
        # if tag=="NULL":
        #     Y_train.append(0)
        # else:
        #     Y_train.append(1)
    Y_test = svm_classifier.predict_proba(X_test)
    
    test_files_list.seek(0)
    for line,num_tag in zip(test_files_list.readlines(),Y_test):
        file = line.strip()
        # if num_tag==1:
        #     output_file.write("1\n")
        # else:
        #     output_file.write("0\n")
        output_file.write(str(num_tag[1])+"\n")
    output_file.close()
    


