import argparse
import cv2
import csv
import numpy as np
from scipy.cluster.vq import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "index file") #path to index file where BOWs are stored
ap.add_argument("-q", "--query", required = True, help = "query image") #path to query image
ap.add_argument("-r", "--result-path", required = True, help = "result path") #path to image dataset folder
ap.add_argument("-ds", "--descriptor", required = True, help = "computed descriptor")	#name of descriptor (sift, surf)
ap.add_argument("-c", "--cl-cnt", required = True, help = "number of clusters") #number of clusters
ap.add_argument("-cs", "--clusters-file", required = True, help = "file of clusters") #path to file where clusters are stored


args = vars(ap.parse_args())

# load the query image and describe it
query = cv2.imread(args["query"])
query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
descriptor_name = args["descriptor"]
k = int(args["cl_cnt"])
ctr_clusters = args["clusters_file"]

#extract descriptors
if descriptor_name in ["surf", "SURF"]:
	extractor = cv2.xfeatures2d.SURF_create()
	dimension=64
elif descriptor_name in ["sift", "SIFT"]:
	extractor = cv2.xfeatures2d.SIFT_create()
	dimension=128
kp, dsc= extractor.detectAndCompute(query, None)
imageCount = 0

with open (ctr_clusters) as f:
	# initialize the CSV reader
	cl_reader = csv.reader(f)

	#read cluster centers from file	 and stack them to numpy array
	centers = np.zeros((1,dimension),"float32")
	for row in cl_reader:
		vis_word = [float(x) for x in row]
		centers=np.vstack((centers, vis_word))
		imageCount = imageCount+1;
	centers = np.delete(centers, (0), axis = 0)
	
	#compute histogram for query image
	query_hstm = np.zeros(k, "float32")	
	words, distance = vq(dsc,centers)
	for w in words:
		query_hstm[w] += 1
	print (query_hstm)

	f.close()

indexPath = (args["index"])
results = {}
with open(indexPath) as fl:
	reader = csv.reader(fl)
	# loop over the rows in the index
	for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
		hstm = [float(x) for x in row[1:]]
		eps = 1e-10	
		d = np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(hstm, query_hstm)])
				# now that we have the distance between the two feature
				# vectors, we can udpate the results dictionary -- the
				# key is the current image ID in the index and the
				# value is the distance we just computed, representing
				# how 'similar' the image in the index is to our query
		results[row[0]] = d

	# close the reader
	fl.close()

	# sort our results, so that the smaller distances (i.e. the
	# more relevant images are at the front of the list)
results = sorted([(v, m) for (m, v) in results.items()])

# show "limit"-number of results
limit = 20
results = results[:limit]
cv2.imshow("Query", query)
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread(args["result_path"] + "/" + resultID)
	cv2.imshow("Result", result)
	cv2.waitKey(0)

	
