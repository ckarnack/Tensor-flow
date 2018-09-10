from __future__ import print_function
import functools
import io
import sys
import os
import time
import csv
import cv2
import numpy as np
from scipy.cluster.vq import *
from scipy.spatial import distance
from pyspark.mllib.clustering import KMeansModel
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row


def search_similar(row, query_bow):
	
	#get image name and bag of words from ROW
    image_name = row['fileName']
    bow = np.array(row['bow'])

    eps = 1e-10
	#compute distance between query bow and image of dataset bow
    d = np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(bow, query_bow)])
                              
    return [(image_name, d)]

if __name__ == "__main__":words
    sc = SparkContext(appName="kmeans_assign")
    sqlContext = SQLContext(sc)

    try:
        query_path = sys.argv[1]
        kmeans_model_path = sys.argv[2]
        bow_parquet_path = sys.argv[3]
        dictionary_size = sys.argv[4]
    except:
        print("not all parameters chosen")
#load dictionary and kmeans model
    bows = sqlContext.read.parquet(bow_parquet_path)
    vocabulary = KMeansModel.load(sc, kmeans_model_path)
    clusterCenters = vocabulary.clusterCenters
#compute query descriptors
    query = cv2.imread(query_path)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    kp, dsc= surf.detectAndCompute(query, None)
#quantize query descriptors
    centers = clusterCenters
    centers = np.asarray(centers)
    words, distance = vq(dsc, centers)
    query_bow = np.zeros(int(dictionary_size), "float32")
    for w in words:
         query_bow[w] += 1
    clusterCenters = sc.broadcast(clusterCenters)

    search_results = bows.rdd.map(functools.partial(search_similar, query_bow=query_bow))
  #  sorted_res = search_results.sortBy(lambda x: x[0][1])
  #  print(sorted_res.take(20))
  #  print("--- %s seconds ---" % (time.time() - start_time))  
   
    sc.stop()
