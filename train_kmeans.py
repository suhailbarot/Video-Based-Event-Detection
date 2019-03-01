#!/bin/python 

import numpy as np
import os
from sklearn.cluster.k_means_ import MiniBatchKMeans,KMeans
import cPickle
import sys
import csv
from tqdm import tqdm
import glob


def get_batch(data):
	batch_size=10000
	for i in xrange(0,len(data),batch_size):
		if i+batch_size<len(data):
			yield data[i:i+batch_size]
		else:
			yield data[i:len(data)]

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
	print "e"
	if len(sys.argv) != 2:
		print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
		print "mfcc_csv_file -- path to the mfcc csv file"
		print "cluster_num -- number of cluster"
		print "output_file -- path to save the k-means model"
		exit(1)
	files = glob.glob('surf/*.npz')
	cluster_num = int(sys.argv[1])
	data = np.ones((20,20),np.float16)

	output_file = 'kmeans_model_'+str(cluster_num)
	kmeans = MiniBatchKMeans(n_clusters=cluster_num)
	for f_name in tqdm(files):
		data = []
		arr = np.load(f_name)['arr_0']
		i=0
		if arr is not None:
			for row in arr:
				if i%10==0:
					data.append(row)
				i+=1
		print(len(data))
		if len(data)>500:
			kmeans = kmeans.partial_fit(data)
	with open(output_file+'.pkl', 'wb') as fid:
		cPickle.dump(kmeans, fid)


	print "K-means trained successfully!"
