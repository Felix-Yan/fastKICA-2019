import numpy as np

def meanMatrix(V):
	Ns = len(V[0,:])
	#Remove mean
	#To take the mean of each row, choose axis = 1.Shape is (n_sources,)
	meanValue = np.mean(V, axis = 1)
	#This changes meanValue from 1d to 2d, now a column vector with size dimension*1.Shape is (n_sources,1)
	meanValue = np.reshape(meanValue,(len(meanValue),1))
	#This creates an array full of ones with the same length as the column number of V
	oneArray = np.ones((1,Ns))
	#This creates a matrix full of mean values for each row
	meanMatrix = np.dot(meanValue,oneArray)

	return meanMatrix

#Mean subtraction. V is data matrix. Number of rows indicates the dimensionality.
def mean_subtraction(V):
	#This creates a matrix full of mean values for each row
	meanMatrix = meanMatrix(V)
	#This gives V zero mean
	V = V - meanMatrix

	return V

## TODO whitening has problem. Debug
def whitening(V):
	#whitening
	n_sources = len(V)
	#this computes the covariance matrix of V. Each row should be a variable and each column should be an observation.
	covMatrix = np.cov(V)
	#debug
	print('Covariance matrix:')
	print(covMatrix)
	#this gets the svd form of the covMatrix.
	P,d,Qt = np.linalg.svd(covMatrix, full_matrices=False)
	Q = Qt.T
	#this gets the first L entries
	d = d[:n_sources]
	D = np.diag(d)
	#this gets the first L columns of singular (eigen) vectors
	E = P[:,:n_sources]
	#this computes the whitening matrix D^(-1/2)*E.T
	whiteningMatrix = np.dot(np.linalg.inv(np.sqrt(D)),E.T)
	#whitened is the whitened signal matrix
	whitened = np.dot(whiteningMatrix,V)

	return whitened, whiteningMatrix