import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import matplotlib.ticker
import time
import os
import sys
###############################################################################################################################################################################################################
# fig parameters
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
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
###############################################################################################################################################################################################################
#deviation from nonlinearity
def NL(A,B,AB):
	return (A+B)/AB - 1

def stdNL(NL):
	return [np.std(NL[i])/np.sqrt(len(NL[i])) for i in range(len(NL))]
###############################################################################################################################################################################################################
#symlog recal
def ySymLog(y,C):
	if y>0:
		if np.log10(abs(y))-C>0:
			return np.sign(y)* np.log10(abs(y))-C
		else:
			return 0
	if y<0:
		if -np.log10(abs(y))+C<0:
			return np.sign(y)* np.log10(abs(y))+C
		else:
			return 0
#x,y ticks
C    = -4
ySymLog_ticks = np.concatenate((np.arange(-1,C-1,-1),np.arange(1,-C+1,1)))
y_tick        = np.concatenate((np.array([r"$-10^{"+str(num)+"}$" for num in range(C+1,0+1,+1)]),np.array([r"$10^{"+str(num)+"}$" for num in range(C+1,0+1,+1)])))
x_tick        = np.array([r"$10^{"+str(num)+"}$" for num in range(0,8,+1)])
###############################################################################################################################################################################################################
# import data AQRH
AQRH = np.genfromtxt('AQRH.txt')
num   = 41
rep   = 30
Tmeas = 20
xaqrh  	 = []
nlaqrh 	 = []
for i in range(1,num+1,1):
	xaqrh.append(AQRH.T[2][(i-1)*rep:(i)*rep])
	nlaqrh.append(NL(AQRH.T[0][(i-1)*rep:(i)*rep],AQRH.T[1][(i-1)*rep:(i)*rep],AQRH.T[2][(i-1)*rep:(i)*rep]))
# AQRH x,xerr
Xaqrh     = np.mean(xaqrh,axis=1)/Tmeas
Xerraqrh  = [np.std(xaqrh[i]/Tmeas)/np.sqrt(len(xaqrh[i])) for i in range(len(xaqrh))]
# AQRH y,yerr
meanaqrh  = np.mean(nlaqrh,axis=1)
stdaqrh   = stdNL(nlaqrh)
#symlog recal
erraqrh   = [meanaqrh - stdaqrh, meanaqrh + stdaqrh]
NLaqrh    = [ySymLog(i,C) for i in meanaqrh]
NLaqrherr = np.array([np.array([abs(ySymLog(erraqrh[0][i],C)-NLaqrh[i]) for i in range(len(erraqrh[0]))]),np.array([abs(ySymLog(erraqrh[1][i],C)-NLaqrh[i]) for i in range(len(erraqrh[1]))])])
################################################################################################################################################################################
# import data AQ4C
AQ4C = np.genfromtxt('AQ4C.txt')
num   = 40
rep   = 30
Tmeas = 20
xaq4c 	= []
nlaq4c	= []
for i in range(1,num+1,1):
	xaq4c.append(AQ4C.T[2][(i-1)*rep:(i)*rep])
	nlaq4c.append(NL(AQ4C.T[0][(i-1)*rep:(i)*rep],AQ4C.T[1][(i-1)*rep:(i)*rep],AQ4C.T[2][(i-1)*rep:(i)*rep]))
# AQ4C x,xerr
Xaq4c     = np.mean(xaq4c,axis=1)/Tmeas
Xerraq4c  = [np.std(xaq4c[i]/Tmeas)/np.sqrt(len(xaq4c[i])) for i in range(len(xaq4c))]
# AQ4C y,yerr
meanaq4c  = np.mean(nlaq4c,axis=1)
stdaq4c   = stdNL(nlaq4c)
#symlog recal
erraq4c   = [meanaq4c - stdaq4c, meanaq4c + stdaq4c]
NLaq4c    = [ySymLog(i,C) for i in meanaq4c]
NLaq4cerr = np.array([np.array([abs(ySymLog(erraq4c[0][i],C)-NLaq4c[i]) for i in range(len(erraq4c[0]))]),np.array([abs(ySymLog(erraq4c[1][i],C)-NLaq4c[i]) for i in range(len(erraq4c[1]))])])
################################################################################################################################################################################
# import data COUNT
COUNT = np.genfromtxt('COUNT.txt')
num   = 40
rep   = 30
Tmeas = 20
xcount	= []
nlcount	= []
for i in range(1,num+1,1):
	xcount.append(AQ4C.T[2][(i-1)*rep:(i)*rep])
	nlcount.append(NL(COUNT.T[0][(i-1)*rep:(i)*rep],COUNT.T[1][(i-1)*rep:(i)*rep],COUNT.T[2][(i-1)*rep:(i)*rep]))
# COUNT x,xerr
Xcount     = np.mean(xcount,axis=1)/Tmeas
Xerrcount  = [np.std(xcount[i]/Tmeas)/np.sqrt(len(xcount[i])) for i in range(len(xcount))]
# COUNT y,yerr
meancount  = np.mean(nlcount,axis=1)
stdcount   = stdNL(nlcount)
#symlog recal
errcount   = [meancount - stdcount, meancount + stdcount]
NLcount   = [ySymLog(i,C) for i in meancount]
NLcounterr = np.array([np.array([abs(ySymLog(errcount[0][i],C)-NLcount[i]) for i in range(len(errcount[0]))]),np.array([abs(ySymLog(errcount[1][i],C)-NLcount[i]) for i in range(len(errcount[1]))])])
################################################################################################################################################################################
# import data ID120
ID120 = np.genfromtxt('ID120.txt')
num   = 37
rep   = 30
Tmeas = 20
xid120	= []
nlid120	= []
for i in range(1,num+1,1):
	xid120.append(ID120.T[2][(i-1)*rep:(i)*rep])
	nlid120.append(NL(ID120.T[0][(i-1)*rep:(i)*rep],ID120.T[1][(i-1)*rep:(i)*rep],ID120.T[2][(i-1)*rep:(i)*rep]))
# id120 x,xerr
Xid120     = np.mean(xid120,axis=1)/Tmeas
Xerrid120  = [np.std(xid120[i]/Tmeas)/np.sqrt(len(xid120[i])) for i in range(len(xid120))]
# id120 y,yerr
meanid120  = np.mean(nlid120,axis=1)
stdid120   = stdNL(nlid120)
#symlog recal
errid120   = [meanid120 - stdid120, meanid120 + stdid120]
NLid120   = [ySymLog(i,C) for i in meanid120]
NLid120err = np.array([np.array([abs(ySymLog(errid120[0][i],C)-NLid120[i]) for i in range(len(errid120[0]))]),np.array([abs(ySymLog(errid120[1][i],C)-NLid120[i]) for i in range(len(errid120[1]))])])
################################################################################################################################################################################
# fig data
x=[Xaqrh,Xaq4c,Xcount,Xid120]
xerr=[Xerraqrh,Xerraq4c,Xerrcount,Xerrid120]
y=[NLaqrh,NLaq4c,NLcount,NLid120]
yerr=[NLaqrherr,NLaq4cerr,NLcounterr,NLid120err]
################################################################################################################################################################################
#plot
fontsize=24
tickssize=20
yfontsize=24
des=18
fig = plt.figure(figsize=(10, 8))
per1 = 0.1
per2 = 0.025
ymin = 10**-3

ax = fig.add_subplot(221)
ax.errorbar(x[0], y[0], yerr[0], xerr[0], fmt='o', color="black")
ax.set_ylabel(r'$\Delta$',size=fontsize)
ax.set_yticks(ySymLog_ticks)
ax.set_yticklabels(y_tick, fontsize=yfontsize)
ax.set_xscale('log')
ax.set_xticks([1,10,10**2,10**3,10**4,10**5,10**6,10**7])
ax.set_xticklabels(x_tick, fontsize=yfontsize)
ax.set_xlim(min(x[0])- per1*min(x[0]),max(x[0]) + per1*max(x[0]))
ax.set_ylim(ySymLog(ymin,C),max(y[0]) + per2*max(y[0]))
ax.grid(b=True, which='major', color='lightgray', linestyle='--')

ax2 = fig.add_subplot(222)
ax2.errorbar(x[1], y[1], yerr[1], xerr[1], fmt='o', color="black")
ax2.set_xscale('log')
ax2.set_xticks([1,10,10**2,10**3,10**4,10**5,10**6,10**7])
ax2.set_xticklabels(x_tick, fontsize=yfontsize)
ax2.set_yticks(ySymLog_ticks)
ax2.set_yticklabels(y_tick, fontsize=yfontsize)
ax2.set_xlim(min(x[1])- per1*min(x[1]),max(x[1]) + per1*max(x[1]))
ax2.set_ylim(ySymLog(ymin,C),max(y[1]) + per2*max(y[1]))
ax2.grid(b=True, which='major', color='lightgray', linestyle='--')

ax3 = fig.add_subplot(223)
ymin = -10**-2
y1=0.04
y2=-0.04
ax3.errorbar(x[2], y[2], yerr[2], xerr[2], fmt='o', color="black")
ax3.axhline(linewidth=4, color='white')
ax3.axhline(y1, color = 'black', lw=1)
ax3.text(400, -0.075, '$\parallel$', fontsize=des, fontweight='normal', color='black', rotation=-30)
ax3.text(0.6*10**6, -0.075, '$\parallel$', fontsize=des, fontweight='normal', color='black', rotation=-30)
ax3.axhline(y2, color = 'black', lw=1)
ax3.set_ylabel(r'$\Delta$',size=fontsize)
ax3.set_xlabel(r'$R_{AB}^{det}$ [Hz]',size=fontsize)
ax3.set_yticks(ySymLog_ticks)
ax3.set_yticklabels(y_tick, fontsize=yfontsize)
ax3.set_xscale('log')
ax3.set_xticks([1,10,10**2,10**3,10**4,10**5,10**6,10**7])
ax3.set_xticklabels(x_tick, fontsize=yfontsize)
ax3.set_xlim(min(x[2])- per1*min(x[2]),max(x[2]) + per1*max(x[2]))
ax3.set_ylim(ySymLog(ymin,C),max(y[2]) + per2*max(y[2]))
ax3.grid(b=True, which='major', color='lightgray', linestyle='--')

ax4 = fig.add_subplot(224)
ymin = 10**-2
ax4.errorbar(x[3], y[3], yerr[3], xerr[3], fmt='o', color="black")
ax4.set_xlabel(r'$R_{AB}^{det}$ [Hz]',size=fontsize)
ax4.set_yticks(ySymLog_ticks)
ax4.set_yticklabels(y_tick, fontsize=yfontsize)
ax4.set_xscale('log')
ax4.set_xticks([1,10,10**2,10**3,10**4,10**5,10**6,10**7])
ax4.set_xticklabels(x_tick, fontsize=yfontsize)
ax4.set_xlim(min(x[3])- per1*min(x[3]),1.01*10**6)
ax4.set_ylim(ySymLog(ymin,C),max(y[3]) + per2*max(y[3]))
ax4.grid(b=True, which='major', color='lightgray', linestyle='--')

for i, label in enumerate(('(a)', '(b)', '(c)', '(d)')):
	ax00 = fig.add_subplot(2,2,i+1)
	if i==1 or i==3:
		ax00.text(-0.25, 1.05, label, transform=ax00.transAxes, fontsize=24, fontweight='normal', va='top', ha='right')
	else:
		ax00.text(-0.40, 1.05, label, transform=ax00.transAxes, fontsize=24, fontweight='normal', va='top', ha='right')

plt.tight_layout(pad=0.5, w_pad=1.1, h_pad=0.5)
plt.savefig('fig3.pdf',dpi=300,bbox_inches='tight',pad_inches = 0.01)

plt.show()

time.sleep(1)
