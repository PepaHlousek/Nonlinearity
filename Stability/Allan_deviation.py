#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
##############################################################################################################################################################################################################
# Setup
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("errorbar", capsize=2)
plt.rc('lines', markersize = 4)
plt.rc("text", usetex = False)
##############################################################################################################################################################################################################
#import data
datafiles = ['stability_SLED.txt', 'stability_SLED_NL.txt']

total_time = 12*60*60.

# Copyright 2009 Miroslav Jezek
# (transfered from C to Python in 2014)
def adev(data, dt=1):
    N = data.size
    #totalmean= np.mean(data)
    max_tau = int(N/2 + 1)
    print("max_tau = "+str(max_tau))
    taus = np.array(range(2,max_tau))
    ad = np.zeros(max_tau-2)
    for tau in range(2,max_tau):
        n_bins = int(N/tau)
        means = np.zeros(n_bins)
        #print("tau="+str(tau)+"    n_bins="+str(n_bins))
        for n in range(n_bins):
            means[n] = np.mean(data[n*tau:(n+1)*tau-1])
        #print((np.mean((means[1:n_bins] - means[0:n_bins-1])**2)))
        ad[tau-2] = np.sqrt(0.5 * (np.mean((means[1:n_bins] - means[0:n_bins-1])**2)))
    #return (dt*taus, ad/totalmean)
    return (dt*taus, ad)
        
for datafile in datafiles:

    data = np.loadtxt(datafile, delimiter=',')

    print(datafile)
    print(data.shape)
    N = data.shape[0]
    print("RAW number of samples = "+str(N))
    print(data[::5])

    print("total_time = "+str(total_time))
    data_interval = total_time / data.shape[0]
    print("RAW data_interval = "+str(data_interval))

    totalmean = np.mean(data)
    print("totalmean = "+str(totalmean))
    data1 = data/totalmean
    print(data1[::5])

    # compute and save:
    (taus,ad1) = adev(data1,data_interval)
    print("length of adev = "+str(ad1.size))

    # plot:
    plt.figure()
    plt.plot(taus, ad1, 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(10**-6,3*10**-4)
    plt.xlim(1,3*10**4)
    plt.grid(color='grey', linestyle='--', linewidth=1)
    plt.xlabel(r'$T$ [s]')
    plt.ylabel(r'$\sigma_\mathrm{Allan}$')
    plt.tight_layout()
    plt.show()
