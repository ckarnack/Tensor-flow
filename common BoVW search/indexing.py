import argparse
import os
import glob
import cv2
import time
import numpy as np
from scipy.cluster.vq import *

def compute_features(dataset, descriptor_name):
	des_list = []	
	imageCount = 0
	for imagePath in glob.glob(dataset + "/*.jpg"):
		# extract the image ID (i.e. the unique filename) from the image
		# path and load the image itself
		imageID = imagePath[imagePath.rfind("/") + 1:]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		imageCount=imageCount+1;
		
		#choosing and computing descriptor
		if descriptor_name in ["surf", "SURF"]:
               		extractor = cv2.xfeatures2d.SURF_create()
		elif descriptor_name in ["sift", "SIFT"]:
               		extractor = cv2.xfeatures2d.SIFT_create()
		kp, dsc= extractor.detectAndCompute(image, None)
		
		# append to des_list		
		des_list.append((imageID, dsc))	   	
	descriptors = des_list[0][1]
	# Stacking all the descriptors into 1 numpy array
	for imageID, descriptor in des_list[1:]:	
		descriptors = np.vstack((descriptors, descriptor))  
	return des_list, descriptors, imageCount	

def create_vocabulary(descriptors, ctr_clusters, k):
	
	#kmeans finishing criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS #kmeans start positioning of centers
	compactness,labels,centers = cv2.kmeans(descriptors,k,None,criteria,10,flags)
	
	#write cluster centers to file
	center = []
	for i in range(k):
		center=centers[i].tolist()
		center = [str(f) for f in center]
		ctr_clusters.write("%s\n" % "," .join(center))
	return centers

def quantizing(des_list, centers, output):
	bow = np.zeros((imageCount, k), "float32")
	#for descriptors of i-image^ for each descriptor in des_list[i][1] get it cluster center and save it to words
	for i in range(imageCount):
		words, distance = vq(des_list[i][1],centers)
		for w in words:
			bow[i][w] += 1
	return bow

if __name__ == "__main__":
	start_time = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required = True, #path to image dataset
	help = "images directory")
	ap.add_argument("-i", "--index", required = True, #path to index file to store BOWs
	help = "index file")
	ap.add_argument("-ds", "--descriptor", required = True, #name of descriptor (sift, surf)
	help = "computed descriptor")	
	ap.add_argument("-c", "--cl-cnt", required = True, #number of clusters
	help = "number of clusters")
	ap.add_argument("-cs", "--clusters-file", required = True, #path to file where clusters are to be stored
	help = "number of clusters")
	args = vars(ap.parse_args())
	output = open(args["index"], "w")
	dataset = args["dataset"]
	descriptor_name = args["descriptor"]
	k = int(args["cl_cnt"])
	ctr_clusters = open(args["clusters_file"], "w")
	
	#compute features
	des_list, descriptors, imageCount = compute_features(dataset, descriptor_name)
	
	#start clusterization
	centers = create_vocabulary(descriptors, ctr_clusters, k)
	
	#get all the BOWs
	bow = quantizing(des_list, centers, imageCount)
	
	#write BOWs to file
	hist = []
	for i in range(imageCount):
		hist=bow[i].tolist()
		hist = [str(f) for f in hist]
		output.write("%s,%s\n" % (des_list[i][0], ",".join(hist)))
	output.close()
	print("--- %s seconds ---" % (time.time() - start_time)) 



