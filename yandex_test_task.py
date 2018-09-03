
from __future__ import division
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import norm
	
#load data from file
data = pd.read_csv("D:/log3.txt", delimiter=';',  names = ["N", "type", "from_mobile"])

#group data by request type and aggregate by 'from_mobile' value,  
stats = data.groupby(['type']).agg({'from_mobile': [np.mean, np.sum, np.size]})

#take parts of requests from mobile for each request type 
p_index = stats['from_mobile']['mean']['/index']
p_home = stats['from_mobile']['mean']['/home']
p_test = stats['from_mobile']['mean']['/test']

print 'part of /index requests from mobile gadgets is {:.3f}'.format(p_index)
print 'part of /home requests from mobile gadgets is {:.3f}'.format(p_home)
print 'part of /test requests from mobile gadgets is {:.3f}'.format(p_test)

#take total number of requests for each request type
n_index = stats['from_mobile']['size']['/index']
n_home = stats['from_mobile']['size']['/home']
n_test = stats['from_mobile']['size']['/test']

#func for calculating confidence interval
def conf_interval(value, p, n):
	var = p*(1-p)/n
	sigma = sqrt(var)
	return norm.interval(value, loc=p, scale=sigma)

conf_int_index = conf_interval(0.95, p_index, n_index)
conf_int_home = conf_interval(0.95, p_home, n_home)
conf_int_test = conf_interval(0.95, p_test, n_test)

print 'confidence interval of 95% for /index request is {:.3f}-{:.3f}'.format(conf_int_index[0], conf_int_index[1])
print 'confidence interval of 95% for /home request is {:.3f}-{:.3f}'.format(conf_int_home[0], conf_int_home[1])
print 'confidence interval of 95% for /test request is {:.3f}-{:.3f}'.format(conf_int_test[0], conf_int_test[1])


#take number of requests from mobile gadgets for each request type
m_index = stats['from_mobile']['sum']['/index']
m_test = stats['from_mobile']['sum']['/test'] 

#func for calculating Z_score
def Z_score(m1, n1, p1, m2, n2, p2):
	p_score = (m1+m2)/(n1+n2)
	M = p1 - p2
	D = p_score*(1-p_score)*(n1+n2)/(n1*n2)
	Z = M/sqrt(D)
	return abs(Z)

Z_a = 1.6449 #Z score table value for confidence level of 5%

Z = Z_score(m_index, n_index, p_index, m_test, n_test, p_test)

if Z<Z_a:
    print "{:.4f} is less than {} so the hypothesus of P_index and P_test equity at confidence level of 5% is true".format(Z, Z_a)
else:
    print "{:.4f} is more than {} so the hypothesus of P_index and P_test equity at confidence level of 5% is false".format(Z, Z_a)

