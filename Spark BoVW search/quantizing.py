from __future__ import print_function
import functools
import io
import sys
import os
import time
import numpy as np
from scipy.spatial import distance
from pyspark.mllib.clustering import KMeansModel
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row


def quantizing(row, centers):
	#get image name and image features from dataframe record
    image_name = row['fileName']
    feature_matrix = np.array(row['features'])
    
	#make numpy array of zeros and get numpy array of cluster centers
    centers = centers.value
    bow = np.zeros(len(centers))
    centers = np.asarray(centers)
   
    #for each element in feature_matrix get it cluster center and save it to words
    words, distance = vq(feature_matrix, centers)
    
	#for each cluster center (w) in words add 1 to bow array
	for w in words:
         bow[w] += 1
    return Row(fileName=image_name, bow=bow.tolist())
	
if __name__ == "__main__":
    sc = SparkContext(appName="kmeans_assign")
    sqlContext = SQLContext(sc)

    try:
        feature_parquet_path = sys.argv[1]
        kmeans_model_path = sys.argv[2]
        bow_parquet_path = sys.argv[3]
    except:
        print("not all parameters chosen")

	#read features, kmeans model, get cluster centers from model and send it to nodes as a variable (centers)
    features = sqlContext.read.parquet(feature_parquet_path)
    vocabulary = KMeansModel.load(sc, kmeans_model_path)
    centers = vocabulary.clusterCenters
    centers = sc.broadcast(centers)
   
	#map function for quantizing
    bag_of_words = features.rdd.map(functools.partial(quantizing, centers=centers))
	
	featuresSchema = sqlContext.createDataFrame(bag_of_words)
    featuresSchema.registerTempTable("images")
    featuresSchema.write.parquet(bow_parquet_path)
    sc.stop()
