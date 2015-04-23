import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os.path
from UKF import UKF # import ukf module

########## Schwefel's function
def f(x):
# Input x is of array type, each dimension has values in [-512, 512]
# f returns function value evaluated at x, f_min =0.0 at x=420.968746*np.ones(n)
    global NFuncEvals
    NFuncEvals = NFuncEvals + 1
    n = len(x) # number of dimensions
    alpha = 418.982887
    s = 0.0
    for i in range(n):
	s -= x[i]*np.sin(np.sqrt(np.absolute(x[i]))) 
    return (s+alpha*n)

########## MAIN
NFuncEvals = 0 # NFuncEvals holds number of function evaluations

# Measurement data
z = 0.0 # f_min=0 at at x_min = 420.968746*np.ones(n)
m = 1 # number of observation points

# Starting point
x_start = np.array([0.0, 0.0]) 
n = len(x_start) # number of model parameters
# Upper and lower bounds for each parameter
xU = 512.0*np.ones(n)
xL = -512.0*np.ones(n)

# Parameters for the UKF
Nk = 10 			# number of UKF iterations
P0 = 4.0**2*np.identity(n)
Q = 0.4**2*np.identity(n)
R = 0.1**2*np.identity(m)


# Parameter for SA
Nc = 2000 			# number of cycles
na = 1.0			# number of accepted solutions
t0 = 6.0			# initial temperature
te = 0.01			# final temperature
frac = (te/t0)**(1.0/(Nc-1.0)) 	# fractional reduction for the temperature every cycle

# Grid data for plot of the first 2 dimensions
i1 = np.linspace(xL[0], xU[0], 100)
i2 = np.linspace(xL[1], xU[1], 100)
x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)
for i in range(x1m.shape[0]):
    for j in range(x1m.shape[1]):
        fm[i][j] = f([x1m[i][j],x2m[i][j]])
# Create a contour plot
plt.figure()
plt.contourf(x1m, x2m, fm)#,lines)
plt.colorbar()
#plt.ion()
#plt.gray()
plt.plot(x_start[0],x_start[1],'kD',markersize=6, label="staring point")
#plt.draw()
#time.sleep(0.1)
plt.xlabel('x1')
plt.ylabel('x2')

# Handling of program variables
xi = x_start		# xi is the proposed move
xiu = np.zeros(n)	# xiu is the improved point of the proposed move
xc = x_start		# xc stores current solutions
fc = f(xc)		# current objective
fbest = fc
xbest = xc
# Current temperature
t = t0
DeltaE_avg = 0.0 # average energy
# Files to store best moves
file_x = 'best_estimates.txt'
file_f = 'best_objectives.txt'
file_NFuncEvals = 'best_NFuncEvals.txt'
if os.path.isfile(file_x):  # remove file if it already exists
    os.remove(file_x)
if os.path.isfile(file_f): 
    os.remove(file_f)
if os.path.isfile(file_NFuncEvals): 
    os.remove(file_NFuncEvals)
with open(file_x, "a") as f_x:
    f_x.write(str(xc)+'\n')
with open(file_f, "a") as f_f:
    f_f.write(str(fc)+'\n')
with open(file_NFuncEvals, "a") as f_fne:
    f_fne.write(str(NFuncEvals-100**2)+'\n')

np.random.seed(111) # seed random generator, run 1
#np.random.seed(222) # seed random generator, run 2
#np.random.seed(333) # seed random generator, run 3
#np.random.seed(444) # seed random generator, run 4
#np.random.seed(555) # seed random generator, run 5

# Start anneling cycles
for i in range(Nc):
    print 'Cycle: ' + str(i) + ' with temperature: ' + str(t)
    # Generate new trial point and clip to upper and lower bounds between [xL,xU]
    while True:
	u = np.random.uniform(0,1,n)
	y = np.sign(u-0.5)*t*((1 + 1.0/t)**np.abs(2*u-1) - 1)
	xi = xc + y*(xU-xL)
	if (xi > xL).all() & (xi < xU).all():
	    break
    fxi = f(xi)
    DeltaE = abs(fxi-fc)
    if (fxi>fc):
        if (i==0): DeltaE_avg = DeltaE
        #p = math.exp(-DeltaE/(DeltaE_avg * t))	# generate probability of acceptance
        p = np.exp(-DeltaE/(DeltaE_avg * t))	# generate probability of acceptance
        if (np.random.uniform()<p):
            accept = True
        else:
            accept = False
    else:
        accept = True
    if (accept==True):
	# IMPROVE THE ACCEPTED SAMPLE BY THE UKF BEFORE UPDATE
	xiu = xi
	P = P0 # assign initial estimation error covariance
	for ii in range(Nk):
	    xiu, P, sigP, innov = UKF(xiu,P,f,z,Q,R)
	    for e in range(n): # clip to upper and lower bounds
                xiu[e] = max(min(xiu[e],xU[e]),xL[e])
	    fxi = f(xiu)
	plt.plot(xiu[0],xiu[1],marker='*',markersize=6, color='k')
        # Update currently accepted solution
        xc = xiu
        fc = fxi
        na = na + 1.0 # increase number of accepted solutions
        DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na # update everage energy
    print 'Current solution: ' + str(xc) + '; current objective: ' + str(fc)
    if (fc < fbest): # the best mininum found
	fbest = fc
	xbest = xc
    with open(file_x, "a") as f_x:
        f_x.write(str(xbest)+'\n')
    with open(file_f, "a") as f_f:
        f_f.write(str(fbest)+'\n')
    with open(file_NFuncEvals, "a") as f_fne:
    	f_fne.write(str(NFuncEvals-100**2)+'\n')
    # Lower the temperature for next cycle
    t = frac * t


# Post-processes
plt.savefig('final_distribution.png',dpi=600)
file_log = 'run_log.txt'
with open(file_log, "w") as f_log:
    f_log.write('Number of function evaluations = '+str(NFuncEvals-100**2)+'\n')
    f_log.write('Number of UKF runs = '+str(Nk)+'\n')
    f_log.write('Initial temperature = '+str(t0)+'\n')
    f_log.write('End temperature = '+str(te)+'\n')
    f_log.write('Cooling fraction = '+str(frac)+'\n')
    f_log.write('Number of accepted moves ='+str(na)+'\n')
    f_log.write('Acceptance ratio = '+ str(na/Nc)+'\n')
    f_log.write('Cov. of the measurement R = \n')
    f_log.write(str(R)+'\n')
    f_log.write('Cov. of initial estimation error P0 = \n')
    f_log.write(str(P0)+'\n')
