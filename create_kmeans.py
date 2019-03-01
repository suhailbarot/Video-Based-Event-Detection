#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import csv
from tqdm import tqdm
# Generate k-means features for videos; each video is represented by a single vector
import glob
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1] 
    file_list = open(sys.argv[3])
    cluster_num = int(sys.argv[2])


    # load the kmeans model
    data = []
    mfcc_files = set(glob.glob('surf/*.npz'))
    kmeans = cPickle.load(open(kmeans_model+'.pkl',"rb"))
    i=0
    for line in tqdm(file_list.readlines()):
        if "surf/"+line.strip()+'.npz' not in mfcc_files:
            # data.append(np.array([0]*cluster_num))
            feature_bog = np.array([1/float(cluster_num)]*cluster_num)
            outfile = open('kmeans/'+line.strip()+'.txt','w+')
            outfile.write(" ".join([str(x) for x in feature_bog]))
        else:
            temp = []
            feature_bog = [0]*cluster_num
            arr = np.load("surf/"+line.strip()+'.npz')['arr_0']
            if arr.shape[0]>0:
                feature_vector = kmeans.predict(arr)
                for clus in feature_vector:
                    feature_bog[clus]+=1
            else:
                feature_bog = np.array([1/float(cluster_num)]*cluster_num)
            outfile = open('kmeans/'+line.strip()+'.txt','w+')
            s = sum(feature_bog)
            outfile.write(" ".join([str(x/float(s)) for x in feature_bog]))
    # x = np.array(data)
    # data_file = open('data.npy','w+')
    # np.save(data_file,x)


    

    print "K-means features generated successfully!"