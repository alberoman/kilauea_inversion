#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 09:36:44 2021

@author: aroman
"""
import numpy as np
import matplotlib.pyplot as plt
from synthetic_test_lib import *
import pickle


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
#InSar parameters: Using acquisition geometry of the real data
looks = pickle.load(open('looks.pickle','rb'))
xSar = np.linspace(-5,5,50)
ySar = np.linspace(-5,5,50)
X,Y = np.meshgrid(xSar,ySar)

xSar = [X.reshape(1,-1),X.reshape(1,-1)]
ySar = [Y.reshape(1,-1),Y.reshape(1,-1)]
covMat = np.diag(np.ones(np.shape(xSar[0].squeeze())))
invCov = np.linalg.inv(covMat)
invCov = [invCov,invCov]
detCov = np.linalg.det(covMat)
detCov = [detCov,detCov]

tGPS,GPSObs,tTilt,tiltxObs,tiltyObs,ulosObs,Ncycles = Twochambers_syntethic_LF_GPS(xs,ys,ds,xd,yd,dd,VsExp,VdExp,kExp,pspd,R3,conds,condd,
                                                                                          deltaPVs,deltaPVd,
                                                                                          ls,ld,mu,S,rhog,const,xStation,yStation,
                                                                                          xSar,ySar,looks,poisson,lame)
#tiltxObs = np.random.normal(loc = tiltxObs,scale = tiltErr)



constants = [tTilt,tGPS,ls,mu,S,rhog,const,xStation,yStation,
             xSar,ySar,looks,poisson,lame]
#Add noise
observed,bestPar,truth,constants = pickle.load(open('bestPar.pickle','rb'))

truth = np.array([xs,ys,ds,xd,yd,dd,VsExp,VdExp,kExp,pspd,R3,conds,condd,ld,deltaPVs,deltaPVd])



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

GPSBest, tiltxBest, tiltyBest,insarBest = DirectModel(bestPar,constants)



plt.figure(1)
plt.plot(tGPS,GPSObs,'b',tGPS,GPSBest,'r')
plt.figure(2)
plt.plot(tTilt,tiltxObs,'b',tTilt,tiltxBest,'r')
plt.figure(3)
plt.plot(tTilt,tiltyObs,'b',tTilt,tiltyBest,'r')
plt.figure(4)
plt.scatter(xSar[0],ySar[0],c=ulosObs[0] - insarBest[0],clim = (0,3.6)),plt.colorbar() 

