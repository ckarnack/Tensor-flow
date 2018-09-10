from __future__ import print_function
import logging
import io
import sys
import os
from PIL import Image

import cv2
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

# feature_extraction function
def extract_features(descriptor_name):

    def extract_features_nested(imgfile_imgbytes):
             image_name, image_bytes = imgfile_imgbytes
			 #restore image from bytes, convert into grayscale and store as numpy array
            
			 image = Image.open(io.BytesIO(image_bytes))
             image = image.convert('L')
             image = np.asarray(image)
			
			#read image from numpy array
             img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			 
			 #choosing descriptor type and extraction
             if descriptor_name in ["surf", "SURF"]:
                extractor = cv2.xfeatures2d.SURF_create()
             elif descriptor_name in ["sift", "SIFT"]:
                extractor = cv2.xfeatures2d.SIFT_create()
             kp, descriptors = extractor.detectAndCompute(img, None)
             return [(image_name, descriptors)] 
    return extract_features_nested


if __name__ == "__main__":
    sc = SparkContext(appName="feature_extractor")
    sqlContext = SQLContext(sc)

    try:
        descriptor_name = sys.argv[1]
        image_seqfile_path = sys.argv[2]
        feature_parquet_path = sys.argv[3]
        partitions = int(sys.argv[4])
    except:
        print("not all parameters chosen>")
	
	#read sequence file to RDD and apply map() to it
    images = sc.sequenceFile(image_seqfile_path, minSplits=partitions)
    features = images.map(extract_features(descriptor_name))
	
	#convert to dataframe format with ROWs
    features = features.map(lambda x: (Row(fileName=x[0][0], features=x[0][1].tolist())))   
    
	#create dataframe and load it to parquet
	featuresSchema = sqlContext.createDataFrame(features)
    featuresSchema.registerTempTable("images")
    featuresSchema.write.parquet(feature_parquet_path)
