import numpy as np

########## Scaled symetric sigma points
def ScaledSymSigmaPoints(xEst, PEst, lamb=1.0):
# INPUT
# eEst: current estimate
# PEst: current estimation error cov. matrix 
# *parameters: a set of parameters characterizing the spread of sigma points
# RETURN
# xPts: sigma-points
# wPts: weights
# nPts: number of sigma-points
	xEst = np.array(xEst) # convert to 1darray if xEst is a list
	n = max(xEst.shape) # number of states
	nPts = 2*n + 1
	xPts = np.zeros((n,nPts))
	wPts = np.zeros(nPts)
	
	Psqrt = np.linalg.cholesky((n+lamb)*PEst)	# matrix square root, lower triangular
	#Psqrt = np.linalg.cholesky((n+lamb)*PEst).transpose()	# matrix square root, upper triangular
	
	xPts[:,0] = xEst	# center point
	xPts[:,1:n+1] = np.array([xEst,]*n).transpose() + Psqrt	# points in the negative side to center
	xPts[:,n+1:nPts] = np.array([xEst,]*n).transpose() - Psqrt	# points in the positive side to center
	
	wPts[0] = lamb/(n+lamb)
	wPts[1:nPts] = 0.5*np.ones((1,nPts-1))/(n+lamb)
	
	return xPts, wPts, nPts


########## Unscented Kalman filter
def UKF(xEst,PEst,hfunc,z,Q,R):
# INPUT
# xEst: current estimate
# PEst: current estimation error cov. matrix 
# hfun: obvervation equation
# z: measurement data
# Q, R: cov. matrices for system noise and measurement noise
# xL, xU: lower and upper bounds for x
# RETURN
# xEst_new
# PEst_new
# xSigmaPts
# innov: innovation 
	xEst = np.array(xEst) # convert to 1darray if xEst is a list
	n = max(xEst.shape)	# number of states in state vector
	#m = len(z)	# number of measurements
	m = 1	# number of measurements
	#m = max(z.shape)	# number of measurements
	
	# Assign sigma-points
	lamb = 1.0 # lambda
	xSigmaPts, wSigmaPts, nsp = ScaledSymSigmaPoints(xEst, PEst, lamb)
	
	# Project the sigma-points through nonlinear function
	xPredSigmaPts = np.zeros((n,nsp))
	zPredSigmaPts = np.zeros((m,nsp))
	for i in range(nsp):
		#for j in range(n): # clip the sigma-points in bounds
		#	xPredSigmaPts[j,i] = max(min(xSigmaPts[j,i],xU[j]),xL[j])			# stationary transition
		xPredSigmaPts[:,i] = xSigmaPts[:,i]			# stationary transition
		zPredSigmaPts[:,i] = hfunc(xPredSigmaPts[:,i]) 		# nonlinear observation function hfunc
	# Apprimate mean
	xPred = np.zeros(n)
	zPred = np.zeros(m)
	for i in range(nsp):
		xPred += wSigmaPts[i]*xPredSigmaPts[:,i]
		zPred += wSigmaPts[i]*zPredSigmaPts[:,i]
	
	# Approximate covariances and cross-covariances
	PPred = np.zeros((n,n))
	#PPred = PEst # testing, 10 Mar 2015
	PxzPred = np.zeros((n,m))
	S = np.zeros((m,m))
	for i in range(nsp):
		PPred   += wSigmaPts[i] * np.outer((xPredSigmaPts[:,i]-xPred),(xPredSigmaPts[:,i]-xPred))
		PxzPred += wSigmaPts[i] * np.outer((xPredSigmaPts[:,i]-xPred),(zPredSigmaPts[:,i]-zPred))
		S       += wSigmaPts[i] * np.outer((zPredSigmaPts[:,i]-zPred),(zPredSigmaPts[:,i]-zPred))
	PPred = PPred + Q
	S     = S + R
	
	# Measurement update
	K = np.dot(PxzPred,np.linalg.inv(S))		# Kalman gain
	innov = z - zPred			# innovation
	xEst = xPred + (np.dot(K,innov)).ravel()	# posterior estimate
	PEst = PPred - np.dot(K,np.dot(S,K.transpose()))# posterior est. err. cov.
	
	return xEst, PEst, xSigmaPts, innov		# xEst to 1D array
