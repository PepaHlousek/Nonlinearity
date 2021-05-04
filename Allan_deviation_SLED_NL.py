#!/usr/bin/env python
#-*- coding: utf-8 -*-
#import numpypy
import numpy as np
import matplotlib.pyplot as plt
import time
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
inch = 25.4  # mm
colwidth = 7.0  # inch #90/inch #mm
##############################################################################################################################################################################################################
datafile = 'stabilita_SLED_NL.txt'
total_time = 4*60*60.

data = np.loadtxt(datafile, delimiter=',')

print(datafile)
print(data.shape)
N = data.shape[0]
#N = 2000
print("RAW number of samples = "+str(N))
#print(data[0][0])
#print(data[data.shape[0]-1][0])
print(data[::5])

print("total_time = "+str(total_time))
data_interval = total_time / data.shape[0]
print("RAW data_interval = "+str(data_interval))

totalmean = np.mean(data)
print("totalmean = "+str(totalmean))
data1 = data/totalmean
print(data1[::5])
#np.savetxt("data.csv", data1, delimiter=',')


# Copyright 2009 Miroslav Jezek
# (transfered from C to Python in 2014)
def adev(data, dt=1):
    N = data.size
    #totalmean= np.mean(data)
    max_tau = N/2 + 1
    print("max_tau = "+str(max_tau))
    taus = np.array(range(2,max_tau),float)
    ad = np.zeros(max_tau-2, float)
    for tau in range(2,max_tau):
        n_bins = N/tau
        means = np.zeros(n_bins, float)
        #print("tau="+str(tau)+"    n_bins="+str(n_bins))
        for n in range(n_bins):
            means[n] = np.mean(data[n*tau:(n+1)*tau-1])
        #print((np.mean((means[1:n_bins] - means[0:n_bins-1])**2)))
        ad[tau-2] = np.sqrt(0.5 * (np.mean((means[1:n_bins] - means[0:n_bins-1])**2)))
    #return (dt*taus, ad/totalmean)
    return (dt*taus, ad)


# compute and save:
(taus,ad1) = adev(data1,data_interval)
print("length of adev = "+str(ad1.size))
np.savetxt("allan.csv", zip(taus,ad1), delimiter=',')

# plot:
plt.figure()
plt.plot(taus, ad1, 'black')
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-6,3*10**-4)
plt.xlim(1,3*10**4)
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.savefig('Allan_SLED_NL.pdf',dpi=300,bbox_inches='tight',pad_inches = 0.01)

# subplot:
plt.figure()
plt.plot(taus, ad1, 'black')
#plt.xscale('log')
plt.yscale('log')
plt.xlim(0,500)
plt.ylim(6*10**-6,4*10**-5)
plt.yticks([6*10**-6,8*10**-6,1*10**-5,2*10**-5,4*10**-5], ['6','8','10','12','14'])
#plt.yticks([2*10**-6,4*10**-6,6*10**-6,8*10**-6,1*10**-5,2*10**-5,4*10**-5], ['2','4','6','8','10','12','14'])
plt.grid(which='major',color='grey', linestyle='--', linewidth=1)
plt.savefig('Allan_SLED_NL_zoom.pdf',dpi=300,bbox_inches='tight',pad_inches = 0.01)

#plt.draw()
#plt.show()
