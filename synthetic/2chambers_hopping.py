#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:37:06 2021
TODO!!!!
Correct the looks for the real cases
Correct the det of the covariance, which should normalize the loglikelihood
Add noise and divide by variance in the lileklihood. Removin to understand how it impacts the solution
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
def log_likelihood(param,
                   tTilt,tGPS,GPS,tiltx,tilty,ulosObs,
                   ls,ld,mu,
                   S,rhog,const,poisson,lame,xStation,yStation,xSar,ySar,tiltErr,invCov,detCov):
    
    xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsSamp,VdSamp,kSamp,pspdSamp,R3Samp,condsSamp, conddSamp,deltapVsSamp,deltapVdSamp = param
    try:
        GPSMod,tiltxMod,tiltyMod,ulosMod = DirectModel(tTilt,tGPS,
                                  xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                  VsSamp,VdSamp,kSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                                  deltapVsSamp,deltapVdSamp,
                                  ls,ld,mu,S,rhog,const,xStation,yStation,xSar,ySar,looks,poisson,lame)
        
        sigma2GPS = GPSErr ** 2
        sigma2tilt = tiltErr ** 2
        likeGPS = (np.sum((GPSObs - GPSMod) ** 2)/sigma2GPS) / len(GPSObs)
        liketx = 0
        likety = 0
        for i in range(np.shape(tiltxObs)[1]):
            liketx = liketx + (np.sum((tiltx[:,i] - tiltxMod[:,i]) ** 2 / sigma2tilt )) / np.shape(tiltxObs)[0]
            likety = likety + (np.sum((tilty[:,i] - tiltyMod[:,i]) ** 2 / sigma2tilt )) / np.shape(tiltyObs)[0]
        likeinSAR = insar_loglike(ulosMod,ulosObs,invCov,detCov)
        like = liketx + likety + likeinSAR + likeGPS 
    except:
        like = 100

    return like

def print_like(param,
                   tTilt,tGPS,GPS,tiltx,tilty,ulosObs,
                   ls,ld,mu,
                   S,rhog,const,poisson,lame,xStation,yStation,xSar,ySar,tiltErr,invCov,detCov):
    
    xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsSamp,VdSamp,kSamp,pspdSamp,R3Samp,condsSamp, conddSamp,deltapVsSamp,deltapVdSamp = param
    try:
        GPSMod,tiltxMod,tiltyMod,ulosMod = DirectModel(tTilt,tGPS,
                                  xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                                  VsSamp,VdSamp,kSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                                  deltapVsSamp,deltapVdSamp,
                                  ls,ld,mu,S,rhog,const,xStation,yStation,xSar,ySar,looks,poisson,lame)
        
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
    except:
        like = 100

    print('like ',like, 'tiltx: ',liketx, 'tilty: ',likety, 'InSAR: ',likeinSAR, ' GPS: ',likeGPS )
    return 

def save_minimum(x,f,accept):
    
    f_old = 1e+10
    try:
        bestPar_old,f_old = pickle.load(open('bestPar.pickle','rb'))
    except:
        pickle.dump((x,f),open('bestPar.pickle','wb'))
    if f < f_old:
        pickle.dump((x,f),open('bestPar.pickle','wb'))
        print_like(x,tTiltObs,tGPSObs,GPSObs,tiltxObs,tiltyObs,ulosObs,ls,ld,mu,S,rhog,const,poisson,lame,
                                           xStation,yStation,xSar,ySar,tiltErr,invCov,detCov)



    
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
ld = 2.5 #in km
condd = 4
conds = 4
#Sources volumes and compressibilities
Vd = 4
Vs = 10
k = 1

VdExp = np.log10(Vd * 1e+9)
VsExp = np.log10(Vs * 1e+9)
kExp = np.log10(k * 1e+9)
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
boundmaxBasin = [xs+2,ys+2,ds+1,xd+2,yd+2,dd+1,11,10,10,11,10,10,10,5,5]
boundminBasin = [xs-2,ys-2,ds-1,xd-2,yd-2,dd-1,1,8,8,0.1,1,1,1,-15,-15]
logpar = [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0]


tGPSObs,GPSObs,tTiltObs,tiltxObs,tiltyObs,ulosObs,Ncycles = Twochambers_syntethic_LF_GPS(xs,ys,ds,xd,yd,dd,VsExp,VdExp,kExp,pspd,R3,conds,condd,
                                                                                          deltaPVs,deltaPVd,
                                                                                          ls,ld,mu,S,rhog,const,xStation,yStation,
                                                                                          xSar,ySar,looks,poisson,lame)
#Add noise
tiltxObs = np.random.normal(loc = tiltxObs,scale = tiltErr)
tiltyObs = np.random.normal(loc = tiltyObs,scale = tiltErr)
GPSObs = np.random.normal(loc = GPSObs,scale = GPSErr)
for i in range(len(xSar)):
    ulosObs[i] = np.random.multivariate_normal(ulosObs[i],covMat)

print('Number of cycles is ', Ncycles)
print('The length of the model is ', len(tGPSObs))
GPSCheck, tiltxCheck, tiltyCheck,insarCheck = DirectModel(tTiltObs,tGPSObs,
                        xs,ys,ds,xd,yd,dd,
                        VsExp,VdExp,kExp,pspd,R3,conds,condd,
                        deltaPVs,deltaPVd,
                        ls,ld,mu,S,rhog,const,xStation,yStation,
                        xSar,ySar,looks,poisson,lame)



truth = np.array([xs,ys,ds,xd,yd,dd,VsExp,VdExp,kExp,pspd,R3,conds,condd,deltaPVs,deltaPVd])
guess = np.zeros(len(truth))
guess = np.zeros(len(truth))
myboundsStep = myBounds(boundmaxBasin,boundminBasin)
for i in range(len(guess)):
    if logpar[i] == 0:
        guess[i] = np.random.normal(loc = truth[i],scale = np.abs(truth[i])*2e-1)
    else:
        val = np.random.normal(loc = 10**truth[i],scale = np.abs(10**truth[i])*2e-1)
        guess[i] = np.log10(val)
minimizer_kwargs = {'method' : 'nelder-mead',
                    'options':{'xtol': 1e-2,'ftol': 1e-2,'maxfev':15000},
                    'args':((tTiltObs,tGPSObs,GPSObs,tiltxObs,tiltyObs,ulosObs,ls,ld,mu,S,rhog,const,poisson,lame,
                                           xStation,yStation,xSar,ySar,tiltErr,invCov,detCov))}

soln = scipy.optimize.basinhopping(log_likelihood, guess,accept_test = myboundsStep,
                                   minimizer_kwargs=minimizer_kwargs,T= 0.5,stepsize = 0.02,
                                   niter = 5000,disp = True,callback = save_minimum)

        
xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,VsSamp,VdSamp,kSamp,pspdSamp,R3Samp,condsSamp, conddSamp,deltapVsSamp,deltapVdSamp = soln.x
GPSBest,tiltxBest,tiltyBest,insarBest = DirectModel(tTiltObs,tGPSObs,
                        xsSamp,ysSamp,dsSamp,xdSamp,ydSamp,ddSamp,
                        VsSamp,VdSamp,kSamp,pspdSamp,R3Samp,condsSamp,conddSamp,
                        deltapVsSamp,deltapVdSamp,
                        ls,ld,mu,S,rhog,const,xStation,yStation,
                        xSar,ySar,looks,poisson,lame)
plt.figure(1)
plt.plot(tGPSObs,GPSObs,'b',tGPSObs,GPSCheck,'r')
plt.figure(2)
plt.plot(tTiltObs,tiltxObs,'b',tTiltObs,tiltxCheck,'r')
plt.figure(3)
plt.plot(tTiltObs,tiltyObs,'b',tTiltObs,tiltyCheck,'r')


plt.legend(['Generated Data','Routine used in the inversion','Basin Hopping'])
bounds = boundsEmcee(Ncycles,xs,ys,ds,xd,yd,dd)
nwalkers = 32
pos = soln.x + 1e-2 * np.random.randn(nwalkers,len(soln.x))
nwalkers, ndim = pos.shape
if constrained == 'True':
    print('You are running inversions using using CONSTRAINED on the slip and the number of cycles')
elif constrained == 'False':
    print('You are running inversions using using UNIFORM priors for pspd and R3')
else:
    print('ERROR: what shoud I do with priors on pspd and R3')
if locations == 'Uniform':
    print('You are running inversions using using UNIFORM priors on source locations')
elif locations == 'Gaussian':
    print('You are running inversion with GAUSSIAN priors on the source locations')
else:
    print('ERROR: what I should do with priors on source locations')
sampler = emcee.EnsembleSampler(nwalkers, ndim,log_probability, args = (tTiltObs,tGPSObs,GPSObs,tiltxObs,tiltyObs,ulosObs,ls,ld,mu,S,rhog,const,poisson,lame,
                                           xStation,yStation,xSar,ySar,tiltErr,invCov,detCov))
sampler.run_mcmc(pos,1000,progress= True)