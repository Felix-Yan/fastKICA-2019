import soundfile as sf
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

#This method returns a demixing matrix that considers the whitening matrix

#Add back mean
#W = [[27.2614, -33.2637], [33.6309, -1.4574]]
# Y = np.dot(W,V) + np.dot(np.dot(W,meanValue),np.ones((1,Ns)))

def run(observation, n_sources,Ns,rounds):
	X = observation
	#this sets the random seed to a fixed number.
	np.random.seed(10)
	#epsilon is the threshold value
	epsilon = 1e-7
	#B is the store place for estimated demixing w vectors
	B = np.zeros((n_sources,n_sources))
	#randomly intialize demixing matrix W
	W = np.random.rand(n_sources,n_sources)
	WBlack = np.random.rand(n_sources,n_sources)
	#iterations can record the number of iterations to find a demixing vector w
	iterations = np.zeros((1,n_sources))
	#Perform ICA
	for round in range(n_sources):
		#w is a column of W
		w = W[:,round].reshape(n_sources,1)
		#this represents the previous w during the 1000 iterations below
		wOld = np.zeros((n_sources,1))
		
		for i in range(1,rounds):
			# print('begin round')
			#Orthogonalizing projection
			w = w - np.dot(np.dot(B,B.T),w) 
			#normalize w
			w = np.divide(w,np.linalg.norm(w))

			# If it is converged, quit
			if np.linalg.norm(w-wOld) < epsilon or np.linalg.norm(w+wOld) < epsilon:
				#to convert w from shape(2,1) to (2,)
				B[:,round] = w.reshape(n_sources)
				# W[round,:] = np.dot(w.T,whiteningMatrix)
				WBlack[round,:] = w.T
				break
			#update wOld
			wOld = w
			hypTan = np.tanh(np.dot(X.T,w))
			w = np.divide((np.dot(X,hypTan) - np.dot(np.sum(1 - np.square(hypTan)).T, w)), Ns)
			w = np.divide(w,np.linalg.norm(w))
			# print('end round')
		iterations[0,round] = i

	return WBlack.T
