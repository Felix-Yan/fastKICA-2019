import tensorflow as tf
import numpy as np
import time
from tensorflow.python.client import timeline
import cProfile
from scipy.stats.stats import pearsonr   
import itertools
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import preprocessing
import random
from scipy.stats.stats import pearsonr 
from scipy import interpolate
import sys
import matlab.engine
import JADEMethod

# setenv CUDA_VISIBLE_DEVICES 0

# Supports all the experiments. Now supports passed arguments for starting and ending generations

## data is of the form 2*N

def amari_distance(P,Q):
	m = len(P)
	B = np.dot(np.linalg.inv(P),Q)
	B = np.absolute(B)
	sum1 = np.sum(B,axis=0)
	max1 = B.max(0)
	term1 = (np.sum(sum1/max1)-m)/(m-1)/m
	B = B.T
	sum2 = np.sum(B,axis=0)
	max2 = B.max(0)
	term2 = (np.sum(sum2/max2)-m)/(m-1)/m
	distance = (term1 + term2)/2 

	return distance	


num_arg = len(sys.argv)
args = sys.argv
low_bound = 0
upp_bound = 0

if num_arg > 1:
	low_bound = int(args[1])
	upp_bound = int(args[2])


n_sources = 2
batch_size = 250
Ns = 250
energy = 1

print('sample size:',Ns)


distance_list = []
repeat = 1
if upp_bound != 0:
	repeat = upp_bound
#set a seed for selecting distribution paris
random.seed(0)
np.random.seed(0)

print('lower bound:',low_bound)
print('upper bound:',upp_bound)

## loading matrices and data
all_matrices = np.load('matrices_2.npy', mmap_mode='r')

all_data = np.load('2_250.npy', mmap_mode='r')
# all_data = np.load('2_1000.npy', mmap_mode='r')
# all_data = np.load('2_1000_noisy.npy', mmap_mode='r')

eng = matlab.engine.start_matlab()
eng.addpath(r'fastKICA/',nargout=0)
eng.addpath(r'fastKICA/utils',nargout=0)

for i in range(low_bound,repeat):
	print('iteration:',i)

	
	A = all_matrices[i*n_sources:(i*n_sources+n_sources),:]
	print('mixing matrix is:')
	print(A)

	#benchmark test		
	S = all_data[i*n_sources:(i*n_sources+n_sources),:]


	#nomalize S to have unit norm
	normed = (S - S.mean(axis=1)[:,None] ) / S.std(axis=1)[:,None]


	#V is the observed signal mixture.
	V = np.dot(A,normed)

	#Remove mean
	# data = preprocessing.mean_subtraction(V)

	#whitening
	data, whiteningMatrix = preprocessing.whitening(V)
	# data = np.transpose(data) #no transpose for fastKICA

	## no whitening
	# data = np.transpose(V)

	##initial weights generated by running JADE
	mixing,S = JADEMethod.cjade(data)
	Xin = np.real(mixing.T)
	maxiter = 20
	sigma = 1.0
	thresh = 1e-6
	MS = matlab.double(data.tolist())
	Xin = matlab.double(Xin.tolist())

	# weights, XS, hsics = eng.fastkica(MS, Xin, maxiter, sigma, thresh)
	output = eng.fastkica(MS, Xin, maxiter, sigma, thresh,nargout=3)
	# print(output[0])
	weights = np.asarray(output[0])
	# print(weights)

	Q1 = np.linalg.inv(weights.T)
	## with whitening
	Q2 = np.dot(whiteningMatrix,A)

	## without whitening
	# Q2 = A

	# distance = eng.amari_distance(Q1Mat,Q2Mat)*100
	distance = amari_distance(Q1,Q2)*100
	print('distance Number ',i,' is: ', distance)
	distance_list.append(distance)


mean = np.mean(distance_list)
var = np.var(distance_list)
std = np.sqrt(var)

print('final result:')
print(mean)
print(std)
