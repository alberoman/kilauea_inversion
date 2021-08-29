#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:37:06 2021
This code generates synthetic case for caldera collapse and uses basinhopping to estimate the MAP solution.
It inverts simultaneously. GPS, tilt, and insar (different looks) geometry.
It works with perturbation of the true solutions of 20% for the initial guess it finds the MAP in approx 
1000-2000 iterations.  Temperature for basinhopping is set to 0.5, the step is set to 0.02. Minimizer
is Nelder-Mead with accuracy for xtol and ftol = 1e-2.
It leaves also the length of the conduit connecting the two chambers as a free parameter to be estimated.

TODO!!!!
Correct the looks for the real cases. 
Not sure about this looks note. The looks are taken from Paul file,so that they should be correct.
@author: aroman
"""


import matplotlib.pyplot as plt
import numpy as np
from synthetic_test_lib import *
import numpy as np
import scipy
import pickle,os
from multiprocessing import Pool
import os
os.environ["OMP_NUM_THREADS"] = "1" 

class myBounds(object):
    def __init__(self,xmax,xmin):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self,**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin        
def log_likelihood(param):
    try:
        GPSMod,tiltxMod,tiltyMod,ulosMod = DirectModel(param,constants)
        
        sigma2GPS = GPSErr ** 2
        sigma2tilt = tiltErr ** 2
        likeGPS = (np.sum((GPSObs - GPSMod) ** 2)/sigma2GPS) / len(GPSObs)
        liketx = 0
        likety = 0
        for i in range(np.shape(tiltxObs)[1]):
            liketx = liketx + (np.sum((tiltxObs[:,i] - tiltxMod[:,i]) ** 2 / sigma2tilt )) / np.shape(tiltxObs)[0]
            likety = likety + (np.sum((tiltyObs[:,i] - tiltyMod[:,i]) ** 2 / sigma2tilt )) / np.shape(tiltyObs)[0]
        likeinSAR = insar_loglike(ulosMod,ulosObs,invCov,detCov)
        like = liketx + likety + likeinSAR + likeGPS
    except:
        like = 1000        
    return like

def print_like(param):
    xs,ys,ds,xd,yd,dd,Vs,Vd,k,pspd,R3,conds,condd,ld,deltapVs,deltapVd = param
    try:
        GPSMod,tiltxMod,tiltyMod,ulosMod = DirectModel(param,constants)
        sigma2GPS = GPSErr ** 2
        sigma2tilt = tiltErr ** 2
        likeGPS = ( np.sum((GPSObs - GPSMod) ** 2) / sigma2GPS) / len(GPSObs)
        liketx = 0
        likety = 0
        for i in range(np.shape(tiltxObs)[1]):
            liketx = liketx + ( np.sum((tiltx[:,i] - tiltxMod[:,i]) ** 2 / sigma2tilt )) / np.shape(tiltxObs)[0]
            likety = likety + (np.sum((tilty[:,i] - tiltyMod[:,i]) ** 2 / sigma2tilt )) / np.shape(tiltyObs)[0]
        likeinSAR = insar_loglike(ulosMod,ulosObs,invCov,detCov)
        like = liketx + likety + likeinSAR + likeGPS 
        print('like ',like, 'tiltx: ',liketx, 'tilty: ',likety, 'InSAR: ',likeinSAR, ' GPS: ',likeGPS )

    except:
        like = 1000
     
    return 

def save_minimum(x,f,accept):
    observed  = {}
    observed['tiltxObs'] = tiltxObs
    observed['tiltyObs'] = tiltyObs
    observed['GPSObs'] = GPSObs
    observed['ulosObs'] = ulosObs
    observed['tTilt'] = tTilt
    observed['tGPS'] = tGPS
    observed['tiltErr'] = tiltErr
    observed['GPSErr'] = GPSErr
    observed['invCov'] = invCov
    observed['detCov'] = detCov

    pickle.dump(observed,open('observed.pickle','wb'))
    f_old = 1e+10
    try:
        bestPar_old,f_old = pickle.load(open('bestPar.pickle','rb'))
    except:
        pickle.dump((observed,x,truth,constants),open('bestPar.pickle','wb'))
    if f < f_old:
        pickle.dump((observed,x,truth,constants),open('bestPar.pickle','wb'))
        print_like(x)

def init_guess():
    guess = []
    for i in range(len(truth)):
        
        if type(truth[i])==list:
            temp = []
            for j in range(len(truth[i])):
                if logpar[i][j] == 0:
                    temp.append(np.random.normal(loc = truth[i][j],scale = np.abs(truth[i][j])*2e-1))
                else:
                    val = np.random.normal(loc = 10**truth[i][j],scale = np.abs(10**truth[i][j])*2e-1)
                    temp.append(np.log10(val))
            guess.append(temp)
        else:
            if logpar[i] == 0:
                guess.append(np.random.normal(loc = truth[i],scale = np.abs(truth[i])*2e-1))
            else:
                val = np.random.normal(loc = 10**truth[i],scale = np.abs(10**truth[i])*2e-1)
                guess.append(np.log10(val))
    return guess
    

    
#np.random.seed(1234)
try:
    os.remove('bestPar.pickle')
except:
    pass
    
constrained = 'False'
locations = 'Uniform'
rho = 2600
g = 9.8
rhog = rho * g
poisson = 0.25
lame = 1e+9
const = -9. /(4 * 3.14) * (1 - poisson) / lame

#Conduit parameters
ls = 3e+4
ld = 2.5
condd = 4
conds = 4
#Sources volumes and compressibilities
Vd = 4
Vs = 10
km = 1
VdExp = np.log10(Vd * 1e+9)
VsExp = np.log10(Vs * 1e+9)
kmExp = np.log10(km * 1e+9)
deltaPVs = -1
deltaPVd = -1.5
#Pressures and friction6
R3 = 5
pspd = 0.2
mu = 100
#Pressures and friction

#Cylinder parameters
Rcyl = 6.0e+2
S = 3.14 * Rcyl**2 

#
xStation = [4,-1.5]
yStation = [0,1]
xs = 3 
xd = -2
ys = -2
yd = 3
ds = 3
dd = 1
tiltErr = 0.1
sarErr =  0.1
GPSErr = 0.1
#InSar parameters: Using acquisition geometry of the real data
looks = pickle.load(open('looks.pickle','rb'))
xSar = np.linspace(-5,5,50)
ySar = np.linspace(-5,5,50)
X,Y = np.meshgrid(xSar,ySar)

xSar = [X.reshape(1,-1),X.reshape(1,-1)]
ySar = [Y.reshape(1,-1),Y.reshape(1,-1)]
covMat = sarErr**2 * np.diag(np.ones(np.shape(xSar[0].squeeze())))
invCov = np.linalg.inv(covMat)
invCov = [invCov,invCov]
detCov = np.linalg.det(covMat)
detCov = [detCov,detCov]
boundmin,boundmax,logpar = generate_bounds(xs, ys, ds, xd, yd, dd)


tGPS,GPSObs,tTilt,tiltxObs,tiltyObs,ulosObs,Ncycles = Twochambers_syntethic_LF_GPS(xs,ys,ds,xd,yd,dd,VsExp,VdExp,kmExp,pspd,R3,conds,condd,
                                                                                          deltaPVs,deltaPVd,
                                                                                          ls,ld,mu,S,rhog,const,xStation,yStation,
                                                                                          xSar,ySar,looks,poisson,lame)
constants = [tTilt,tGPS,ls,mu,S,rhog,const,xStation,yStation,
             xSar,ySar,looks,poisson,lame]
#Add noise
tiltxObs = np.random.normal(loc = tiltxObs,scale = tiltErr)
tiltyObs = np.random.normal(loc = tiltyObs,scale = tiltErr)
GPSObs = np.random.normal(loc = GPSObs,scale = GPSErr)
for i in range(len(xSar)):
    ulosObs[i] = np.random.multivariate_normal(ulosObs[i],covMat)

print('Number of cycles is ', Ncycles)
print('The length of the model is ', len(tGPS))
truth = [xs,ys,ds,xd,yd,dd,VsExp,VdExp,kmExp,pspd,R3,conds,condd,ld,deltaPVs,deltaPVd]
GPSCheck, tiltxCheck, tiltyCheck,insarCheck = DirectModel(truth,constants)



guess = init_guess()
myboundsStep = myBounds(boundmax,boundmin)

minimizer_kwargs = {'method' : 'nelder-mead',
                    'options':{'xtol': 1e-2,'ftol': 1e-2,'maxfev':15000}}

soln = scipy.optimize.basinhopping(log_likelihood, guess,accept_test = myboundsStep,
                                   minimizer_kwargs=minimizer_kwargs,T= 0.5,stepsize = 0.02,
                                   niter = 5000,disp = True,callback = save_minimum)

        
xs,ys,ds,xd,yd,dd,Vs,Vd,k,pspd,R3,conds, condd,deltapVs,deltapVd = soln.x
GPSBest,tiltxBest,tiltyBest,insarBest = DirectModel(tTilt,tGPS,
                        xs,ys,ds,xd,yd,dd,
                        Vs,Vd,k,pspd,R3,conds,condd,
                        deltapVs,deltapVd,
                        ls,ld,mu,S,rhog,const,xStation,yStation,
                        xSar,ySar,looks,poisson,lame)
plt.figure(1)
plt.plot(tGPS,GPSObs,'b',tGPS,GPSCheck,'r')
plt.figure(2)
plt.plot(tTilt,tiltxObs,'b',tTilt,tiltxCheck,'r')
plt.figure(3)
plt.plot(tTilt,tiltyObs,'b',tTilt,tiltyCheck,'r')
plt.figure(1)
plt.plot(tGPS,GPSBest,'k')
plt.figure(2)
plt.plot(tTilt,tiltxBest,'k')
plt.figure(3)
plt.plot(tTilt,tiltyBest,'k')

plt.legend(['Generated Data','Routine used in the inversion','Basin Hopping'])
