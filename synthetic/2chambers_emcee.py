#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:57:38 2021
This code  uses the MAP solution calculated from 2chambers.py via basinhopping and uses emcee to sample the posterior.
@author: aroman
"""

import matplotlib.pyplot as plt
import numpy as np
from synthetic_test_lib import *
import numpy as np
import scipy
import pickle,os
from multiprocessing import Pool
import emcee
os.environ["OMP_NUM_THREADS"] = "1" 


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

def init_guess():
    guess = []
    for i in range(len(truth)):
        
        if type(truth[i])==list:
            temp = []
            for j in range(len(truth[i])):
                if logpar[i][j] == 0:
                    temp.append(np.random.normal(loc = truth[i][j],scale = np.abs(truth[i][j]) * perturbation))
                else:
                    val = np.random.normal(loc = 10**truth[i][j],scale = np.abs(10**truth[i][j]) * perturbation)
                    temp.append(np.log10(val))
            guess.append(temp)
        else:
            if logpar[i] == 0:
                guess.append(np.random.normal(loc = truth[i],scale = np.abs(truth[i]) * perturbation))
            else:
                val = np.random.normal(loc = 10**truth[i],scale = np.abs(10**truth[i]) * perturbation)
                guess.append(np.log10(val))
    return np.array(guess)

def check_bounds(param):
    checkFlags = []
    for i,p in enumerate(param):
        checkFlags.append(p > boundmin[i] and p < boundmax[i])
    if all(checkFlags):
        return 0.0
    else: 
        return -np.inf
        
def log_probability(param):
    resultBounds = check_bounds(param)
    ll = log_likelihood(param)
    return -ll + resultBounds

perturbation = 1e-3 # Perturbation of the solution given by basinhopping

observed,bestPar,truth,constants = pickle.load(open('bestPar4Emcee.pickle','rb'))

tiltxObs = observed['tiltxObs'] 
tiltyObs = observed['tiltyObs'] 
GPSObs = observed['GPSObs']
ulosObs = observed['ulosObs'] 
tTilt = observed['tTilt'] 
tGPS = observed['tGPS'] 
tiltErr = observed['tiltErr'] 
GPSErr = observed['GPSErr'] 
invCov = observed['invCov'] 
detCov = observed['detCov'] 
xs = truth[0] 
ys = truth[1] 
ds = truth[2]
xd = truth[3]
yd = truth[4]
dd = truth[5] 

boundmin,boundmax,logpar = generate_bounds(xs, ys, ds, xd, yd, dd)


GPSCheck, tiltxCheck, tiltyCheck,insarCheck = DirectModel(bestPar,constants)
plt.figure(1)
plt.plot(tGPS,GPSObs,'b',tGPS,GPSCheck,'r')
plt.figure(2)
plt.plot(tTilt,tiltxObs,'b',tTilt,tiltxCheck,'r')
plt.figure(3)
plt.plot(tTilt,tiltyObs,'b',tTilt,tiltyCheck,'r')

nwalkers = 32
ndim = len(bestPar)
pos  = np.zeros((nwalkers,ndim))
for i in range(nwalkers):
    pos[i,:] = init_guess()

Sampler = emcee.EnsembleSampler(nwalkers, ndim,log_probability)
Sampler.run_mcmc(pos,10000,progress= True)