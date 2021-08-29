#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:35:16 2020

@author: aroman
"""

import numpy as np
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt
import time 

def ps_analytic(t,R3,T1,phi,a,b,c,d,pd0,ps0):
    ps = -R3 + (-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a)
    return  ps
def ps_diff(ps,pd,R3,T1):
    return  pd - ps - T1 * (ps + R3)

def ps_analytic_root(t,R3,T1,phi,a,b,c,d,pd0,ps0,pslip):
    ps_root = -R3 + (-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a) - pslip
    return  ps_root

def pd_analytic(t,R3,T1,phi,a,b,c,d,pd0,ps0):
    pd = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a)
    return pd
def pd_diff(ps,pd, phi):
    return - phi * (pd - ps) 

def pd_analytic_root(t,R3,T1,phi,a,b,c,d,pd0,ps0,pslip):
    ps_root = -R3 + (-c + d)*(-R3 - pd0 + (R3 + ps0)*(T1 + a - phi + 1)/2) * np.exp(t*(b - c))/a + (c + d)*(2*R3 + 2*a*(R3 + ps0) + 2*pd0 - (R3 + ps0)*(T1 + a - phi + 1)) * np.exp(t*(b + c))/(2*a) - pslip
    return  ps_root

def TwoChambers_UF(w0,par,pslip,tslip,tsl_seed):
    ps0,pd0 = w0
    r3,t1,phi,a,b,c,d = par
    #tslip = optimize.newton(ps_analytic_root, tsl_seed, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    try:
        tslip = optimize.brentq(ps_analytic_root,0,1e+7, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    except:
        print('Invalid tslip')
        tslip = 1e+3
    tsegment = np.linspace(0,tslip,30)
    pssegment = ps_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdend = pd_analytic(tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    return tsegment,tslip,pssegment,pdsegment,pdend

def TwoChambers_LF(w0,par,pslip,tslip):
    ps0,pd0 = w0
    r1,r3,r5,t1,phi,a,b,c,d = par
    try:
        tslip = optimize.brentq(pd_analytic_root,0,1e+13, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    except:
        print('Invalid tslip')
        tslip  = 1e+3
    tsegment = np.linspace(0,tslip,50)
    pssegment = ps_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment,r3,t1,phi,a,b,c,d,pd0,ps0)
    psend = pssegment[-1]
    return tsegment,tslip,pssegment,pdsegment,psend

def TwoChambers_LF_timein(w0,par,pslip,tslip,time,ps,pd,t_x,x_data,N,alpha):
    ps0,pd0 = w0
    r1,r3,r5,t1,phi,a,b,c,d = par
    tsegment = time[time >= tslip]
    pssegment = ps_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    pdsegment = pd_analytic(tsegment - tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    ps[time >= tslip] = pssegment
    pd[time >= tslip] = pdsegment
    x_data[t_x >= tslip] =4 * alpha / (1 + r1) * N
    try:
        tslip = optimize.brentq(pd_analytic_root,0,1e+12, maxiter =  5000, args = (r3,t1,phi,a,b,c,d,pd0,ps0,pslip))
    except:
        tslip = 1e+3
    psend = ps_analytic(tslip,r3,t1,phi,a,b,c,d,pd0,ps0)
    return tslip,ps,pd,psend,x_data

def CalcTilt(DeltaP,xSour,ySour,depth,xSt,ySt,V,pois,lam):
    #Calculate the tilt at one station due to a pressure source which is function of time 
    cs = -9. /(4 * 3.14) * (1 - pois) / lam
    tx = cs * V * DeltaP * depth * (xSt -  xSour)  / (depth**2 + (xSt -  xSour)**2 + (ySt -  ySour)**2 )**(5./2)
    ty = cs * V * DeltaP * depth * (ySt -  ySour)  / (depth**2 + (xSt -  xSour)**2 + (ySt -  ySour)**2 )**(5./2)
    return tx,ty

def insar_deformation(params,    #Inversion parameters
                      x, y,UEast,UNorth,UVert,poisson,lame):#Fixed parameter
    '''
    Calculates InSAR deformation for one source given satellite look geometry
    '''
    x  = x * 1000
    y = y * 1000
    elastic_constant = (1 - poisson) / lame
    xSource, ySource, depthSource,strengthSource = params
    R= np.sqrt((x - xSource)**2 + (y - ySource)**2 + depthSource **2)
    #east, north, vert,dv =  libcdm.CDM(x, y, xSource, ySource, depthSource, ax, ay, az, omegaX, omegaY, omegaZ, disp, poisson)
    coeff_east = (x - xSource)/ R**3 * elastic_constant
    coeff_north = (y - ySource)/ R**3 * elastic_constant
    coeff_vert = depthSource / R**3 * elastic_constant
    east = strengthSource * coeff_east
    north = strengthSource * coeff_north
    vert = strengthSource * coeff_vert
    Ulos = east * UEast + north * UNorth + vert * UVert
    return Ulos
    

def insar_nsources(params,
                   x, y,look,poisson,lame,nSources):
    '''
    Combines deformations associated with two sources given one satellite look geometry
    '''
    UEast,UNorth,UVert = look
    UEast = np.mean(UEast)
    UNorth = np.mean(UNorth)
    UVert = np.mean(UVert)
    UlosTot = np.zeros(np.shape(x))
    for i in range(nSources):
        paramsOneCDM = params[i]
        Ulos = insar_deformation(paramsOneCDM,    #Inversion parameters
                      x, y,UEast,UNorth,UVert,poisson,lame)
        UlosTot = UlosTot + Ulos
    return UlosTot

def insar_ascdesc(params,
                  xx,yy,looks,
                  poisson,lame,nSources):
    '''
    Combines deformations associated with two sources given two geometries.
    Arguments: xx,yy are matrices corresponding to the coordinates of InSAR observations.
    The elements correspond to the ascending descending geometry. Looks is a matrix with
    rows corresponding to the 3 components of the satellite vector and the elements
    corresponding to the two geometry
    
    '''
    UlosTot = []
    for i in range(len(xx)):
        Ulos = insar_nsources(params,
                             xx[i],yy[i],looks[i],poisson,lame,nSources)
        UlosTot.append(Ulos.squeeze())
    return UlosTot

def insar_loglike(UlosMod,
                  UlosObs,invCov,detCov):
    '''
    Calculates the loglikelihood of the insar data both ascending  and descending tracks. 
    InvCov is a list containing the inverse of the InSAR covariance matrix for both track.
    detCov is a list containing the determinant of the covariance matrices
    '''
    logLikeSar = 0
    for i in range(len(UlosMod)):
        UlosModi = np.reshape(UlosMod[i],(len(UlosMod[i]),1))
        UlosObsi = np.reshape(UlosObs[i],(len(UlosObs[i]),1))
        invCovi = invCov[i]
        detCovi = detCov[i]
        misfit = np.matmul(np.transpose(UlosObsi - UlosModi),np.matmul(invCovi,UlosObsi - UlosModi )) / len(UlosObsi)
        logLikeSar = logLikeSar + misfit[0,0]
    return logLikeSar

    


def Twochambers_syntethic_LF_GPS(xs,ys,ds,xd,yd,dd,VsExp,VdExp,kmExp,pspd,R3,conds,condd,deltaPVs,deltaPVd,
                                 ls,ld,mu,S,rhog,const,xStation,yStation,xSar,ySar,looks,poisson,lame):
    #LF = Lower Feeder: in this case the piston is collapsing in the chamber t
    #hat is not feeding directly the eruption (it has only one conduit) 
    #CAREFUL: if you want to change this you have to change the parameters for xstar
    #Calculate synthetic timeseries,including GPS with 2 chambers
    # The input parameters are the one we do not want to invert for, are fixed
    #SOurces locations
    nSources = 2
    ld = ld * 1000
    xs = xs * 1000
    xd = xd * 1000
    ys = ys * 1000
    yd = yd * 1000
    ds = ds * 1000
    dd = dd * 1000

    xSource = np.array([xs,xd]) 
    ySource = np.array([ys,yd])  
    depthSource = np.array([ds,dd]) 
    alpha = 1
    Vs = 10**VsExp
    Vd  = 10**VdExp
    km = 10**kmExp
    kcs = 4. / 3. * lame #geometrical incompressibility of the s-chamber
    kcd = 4. / 3. * lame #geometrical incompressibility of the d-chamber
    ks = (kcs * km) / (kcs + km) #bulk incompressibility s-chamber
    kd = (kcd * km) / (kcd + km) #buld incompressibility d-chamber
    pspd = pspd * 1e+6
    R5 =0

    R1 = rhog * Vd /(kd*S)
    T1 = (conds / condd )**4 * ld /ls
    PHI = kd /ks * Vs / Vd

    params = [T1,PHI,R3] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
    tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
    xstar = pspd * Vd / (kd * S)

    #Careful if you want to change to the UF version (this is LF)
    #xstar should be 
    #xstar = taud * Vs / (ks * S**2)
    A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
    B = -T1/2 - PHI/2 - 1./2
    C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
    D = T1/2 - PHI /2 + 1./2
    params = [R1,R3,R5,T1,PHI,A,B,C,D]
    #Original timseries
    #Initialization
    
    PD0 =   4 * alpha /(1 + R1)
    PSLIP = - 4 * alpha * R1 * (1 - R5)/(1 + R1)
    TSLIP = 0
    #tseg,tslip,PS,PD,ps0 = TwoChambers_LF(np.array([0.1,0.1]),params,0,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
    PS0 = 0
    #PS0 = 0
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    N_cycles =int(((1 + R1)/ (4 * alpha * R1) * R3)-1)
    i  = 1
    t_base = []
    ps_base = []
    pd_base = []
    tcounter = 0
    for i in range(1,N_cycles + 1):
        tseg,tslip,PS,PD,ps0 = TwoChambers_LF(w0,params,PSLIP,TSLIP)
        ps_base.append(PS)
        pd_base.append(PD)
        t_base.append(tseg + tcounter)
        PD0 =   + 4 * alpha / (1 + R1) -4 * alpha * R1 * (1 - R5)/(1 + R1) * i
        PS0 = ps0
        PSLIP =  - 4 * alpha * R1 * (1 - R5)/(1 + R1) * (i + 1)
        w0 = np.array([PS0,PD0])
        tcounter = tcounter + tslip

    pd_base = np.concatenate((pd_base)) 
    ps_base = np.concatenate((ps_base)) 
    t_base = np.concatenate((t_base)) 

    #Resempling
    tOrigGPS = np.linspace(0,np.max(t_base),int(N_cycles * 1.5))
    tOrigTilt = np.linspace(0,np.max(t_base),int(N_cycles * 100))

    tcounter = 0

    gps = np.zeros(len(tOrigGPS))
    ps = np.zeros(len(tOrigTilt))
    pd = np.zeros(len(tOrigTilt))
    PD0 =  + 4 * alpha / (1 + R1) 
    PSLIP = - 4 * alpha * R1 * (1 - R5)/(1 + R1)
    TSLIP = 0
    PS0 =  0
    w0 = np.array([PS0,PD0])
    i  = 1
    while i < N_cycles + 1:
        tslip,ps,pd,ps0,gps = TwoChambers_LF_timein(w0,params,PSLIP,TSLIP,tOrigTilt,ps,pd,tOrigGPS,gps,i,alpha)
        PD0 =   + 4 * alpha / (1 + R1) -4 * alpha * R1 * (1 - R5)/(1 + R1) * i
        PS0 = ps0
        PSLIP =   - 4 * alpha * R1 * (1 - R5)/(1 + R1) * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
        i = i + 1
    
    ps = ps * pspd
    pd = pd * pspd
    ps_base = ps_base * pspd
    pd_base = pd_base * pspd
    tOrigTilt = tOrigTilt * tstar
    gps = gps * xstar
    tOrigGPS = tOrigGPS * tstar
    t_base = t_base * tstar
    tx = np.zeros((len(ps),len(xStation)))
    ty = np.zeros((len(ps),len(xStation)))
    for i in range(len(xStation)):
        x = xStation[i] * 1000
        y = xStation[i] * 1000
        coeffxd = const * depthSource[1] * (x -  xSource[1]) / (depthSource[1]**2 + (x -  xSource[1])**2 + (y -  ySource[1])**2 )**(5./2)
        coeffyd = const * depthSource[1] * (y -  ySource[1]) / (depthSource[1]**2 + (x -  xSource[1])**2 + (y -  ySource[1])**2 )**(5./2)
        coeffxs = const * depthSource[0] * (x -  xSource[0]) / (depthSource[0]**2 + (x -  xSource[0])**2 + (y -  ySource[0])**2 )**(5./2)
        coeffys = const * depthSource[0] * (y -  ySource[0]) / (depthSource[0]**2 + (x -  xSource[0])**2 + (y -  ySource[0])**2 )**(5./2)
        tiltx = coeffxd * Vd * pd + coeffxs * Vs *ps
        tilty = coeffyd * Vd * pd + coeffys * Vs * ps
        tx[:,i] = tiltx *1e+6
        ty[:,i] = tilty *1e+6
    deltaPVs = deltaPVs * 1e+6
    deltaPVd = deltaPVd * 1e+6
    parSar = [[xd,yd,dd,deltaPVd * Vd],
              [xs,ys,ds,deltaPVs * Vs]]
    Ulos = insar_ascdesc(parSar, xSar, ySar, looks, poisson, lame, nSources)
    return tOrigGPS,gps,tOrigTilt,tx,ty,Ulos,N_cycles

def DirectModel(dynamicPar,constants):
    #This direct model take only two input. The list of 
    xs,ys,ds,xd,yd,dd,VsExp,VdExp,kmExp,pspd,R3,conds,condd,ld,deltaPVs,deltaPVd = dynamicPar
    tTilt,tGPS,ls,mu,S,rhog,const,xStation,yStation,xSar,ySar,looks,poisson,lame = constants
    nSources =  2
    ld  = ld * 1000
    xs = xs * 1000
    xd = xd * 1000
    ys = ys * 1000
    yd = yd * 1000
    ds = ds * 1000
    dd = dd * 1000
    xSource = np.array([xs,xd]) 
    ySource = np.array([ys,yd]) 
    depthSource = np.array([ds,dd]) 
    alpha = 1
    Vs = 10**VsExp
    Vd  = 10**VdExp
    km = 10**kmExp
    kcs = 4. / 3. * lame #geometrical incompressibility of the s-chamber
    kcd = 4. / 3. * lame #geometrical incompressibility of the d-chamber
    ks = (kcs * km) / (kcs + km) #bulk incompressibility s-chamber
    kd = (kcd * km) / (kcd + km) #buld incompressibility d-chamber
    pspd = pspd * 1e+6
    R5 = 0
   
    R1 = rhog * Vd /(kd*S)
    T1 = (conds / condd )**4 * ld /ls
    PHI = kd /kd * Vs / Vd
    params = [T1,PHI,R3] #R1 is the ratio of the hydraulic parameters (for high values the top conduit is more efficient)
    tstar = Vs * 8 * mu * ld / (ks * 3.14 * condd**4)
    xstar = pspd * Vd / (kd * S)

    #Careful if you want to change to the UF version (this is LF)
    #xstar should be 
    #xstar = taud * Vs / (ks * S**2)
    A = np.sqrt(T1**2 - 2*T1*PHI + 2*T1 + PHI**2 + 2*PHI + 1)
    B = -T1/2 - PHI/2 - 1./2
    C =  np.sqrt(4*PHI + (-T1 + PHI - 1)**2)/2    
    D = T1/2 - PHI /2 + 1./2
    params = [R1,R3,R5,T1,PHI,A,B,C,D]
   
    
    
    tGPS = tGPS /tstar
    tTilt = tTilt / tstar
    ps = np.ones(len(tTilt))
    pd = np.ones(len(tTilt))
    gps = np.ones(len(tGPS))
    PD0 =   4 * alpha /(1 + R1)
    PSLIP = - 4 * alpha * R1 * (1 - R5)/(1 + R1)
    TSLIP = 0
    #tseg,tslip,PS,PD,ps0 = TwoChambers_LF(np.array([0.1,0.1]),params,0,TSLIP,TSLIP_seed) # Calculate ps at the first cycle
    PS0 =  0
    #PS0 = 0
    w0 = np.array([PS0,PD0])
    TSLIP = 0
    N_cycles = int(((1 + R1)/ (4 * alpha * R1) * R3)-1)
    i  = 1
    thresh = 25
    while i < N_cycles + 1 and i < thresh:
        tslip,ps,pd,ps0,gps = TwoChambers_LF_timein(w0,params,PSLIP,TSLIP,tTilt,ps,pd,tGPS,gps,i,alpha)
        PD0 =   + 4 * alpha / (1 + R1) -4 * alpha * R1 * (1 - R5)/(1 + R1) * i
        PS0 = ps0
        PSLIP =   - 4 * alpha * R1 * (1 - R5)/(1 + R1) * (i + 1)
        TSLIP = TSLIP + tslip
        w0 = np.array([PS0,PD0])
        i = i + 1
    ps = ps * pspd
    pd = pd * pspd
    gps = gps * xstar
    tx = np.zeros((len(ps),len(xStation)))
    ty = np.zeros((len(ps),len(xStation)))
    for i in range(len(xStation)):
        x = xStation[i] * 1000
        y = xStation[i] * 1000
        coeffxd = const * depthSource[1] * (x -  xSource[1]) / (depthSource[1]**2 + (x -  xSource[1])**2 + (y -  ySource[1])**2 )**(5./2)
        coeffyd = const * depthSource[1] * (y -  ySource[1]) / (depthSource[1]**2 + (x -  xSource[1])**2 + (y -  ySource[1])**2 )**(5./2)
        coeffxs = const * depthSource[0] * (x -  xSource[0]) / (depthSource[0]**2 + (x -  xSource[0])**2 + (y -  ySource[0])**2 )**(5./2)
        coeffys = const * depthSource[0] * (y -  ySource[0]) / (depthSource[0]**2 + (x -  xSource[0])**2 + (y -  ySource[0])**2 )**(5./2)
        tiltx = coeffxd * Vd * pd + coeffxs * Vs *ps
        tilty = coeffyd * Vd * pd + coeffys * Vs * ps
        tx[:,i] = tiltx * 1e+6
        ty[:,i] = tilty *1e+6


    deltaPVs = deltaPVs * 1e+6
    deltaPVd = deltaPVd * 1e+6
    parSar = [[xd,yd,dd,deltaPVd * Vd],
              [xs,ys,ds,deltaPVs * Vs]]
    
    Ulos = insar_ascdesc(parSar, xSar, ySar, looks, poisson, lame, nSources)
    return gps,tx,ty,Ulos

def generate_bounds(xs,ys,ds,xd,yd,dd):
    boundmax = [xs+2,ys+2,ds+1,xd+2,yd+2,dd+1,11,10,10,11,10,10,10,5,5,5]
    boundmin = [xs-2,ys-2,ds-1,xd-2,yd-2,dd-1,1,8,8,0.1,1,1,1,1,-15,-15]
    logpar = [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0]
    return boundmin,boundmax,logpar


   