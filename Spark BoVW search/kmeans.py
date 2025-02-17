from __future__ import print_function
import io
import sys
import os

import numpy as np
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SQLContext, Row


if __name__ == "__main__":
    sc = SparkContext(appName="kmeans_dictionary_creation")
    sqlContext = SQLContext(sc)

    try:
        k = int(sys.argv[1]) # number of clusters
        feature_parquet_path = sys.argv[2]
        kmeans_model_path = sys.argv[3]
    except:
        print("not all parameters chosen")

    features = sqlContext.read.parquet(feature_parquet_path)

    # flatMap returns every list item as a new row for the RDD
    # hence transforming x, 128 to x rows of 1, 128 in the RDD.
    # This is needed for KMeans.
    features = features.rdd.flatMap(lambda x: x['features']).cache()
    model = KMeans.train(features, k, maxIterations=10, initializationMode="random")

    model.save(sc, kmeans_model_path)
    print("Clusters have been saved as text file to %s" % kmeans_model_path)
    print("Final centers: " + str(model.clusterCenters))
    sc.stop()
