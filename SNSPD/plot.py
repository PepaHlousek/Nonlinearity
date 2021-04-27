import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import matplotlib.ticker
import os
import sys

###############################################################################################################################################################################################################
# fig parameters
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
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
C    		  = -4
ySymLog_ticks = np.concatenate((np.arange(-1,C-1,-1),np.arange(1,-C+2,1)))
y_tick        = np.concatenate((np.array([r"$-10^{"+str(num)+"}$" for num in range(C+1,0+1,+1)]),np.array([r"$10^{"+str(num)+"}$" for num in range(C+1,0+2,+1)])))
x_tick		  = [10**i for i in np.arange(0,9,+1)]
x_ticklabel   = np.array([r"$10^{"+str(num)+"}$" for num in range(0,9,+1)])
###############################################################################################################################################################################################################
# import data fig. 4 (a),(b)
DC=np.genfromtxt("SNSPD_DC.txt")
tau=np.genfromtxt("deadtime.dat")
eff=np.genfromtxt("SNSPD_eff.txt")
effSNSPD=np.genfromtxt("SNSPD_effplot.txt")
spad=np.genfromtxt("spad_eff.txt",delimiter=',')
###############################################################################################################################################################################################################
# import data fig. 4 (c)
# data CH7_i266
i266 = np.genfromtxt('i266.txt')
num   = 22
rep   = 20
Tmeas = 30
xi266  	 = []
nli266 	 = []
for i in range(1,num+1,1):
	xi266.append(i266.T[2][(i-1)*rep:(i)*rep])
	nli266.append(NL(i266.T[0][(i-1)*rep:(i)*rep],i266.T[1][(i-1)*rep:(i)*rep],i266.T[2][(i-1)*rep:(i)*rep]))

# CH7_i266 x,xerr
Xi266     = np.mean(xi266,axis=1)/Tmeas
Xerri266  = [np.std(xi266[i]/Tmeas)/np.sqrt(len(xi266[i])) for i in range(len(xi266))]
# CH7_i266 y,yerr
meani266  = np.mean(nli266,axis=1)
stdi266   = stdNL(nli266)
#symlog recal
erri266   = [meani266 - stdi266, meani266 + stdi266]
NLi266    = [ySymLog(i,C) for i in meani266]
NLi266err = np.array([np.array([abs(ySymLog(erri266[0][i],C)-NLi266[i]) for i in range(len(erri266[0]))]),np.array([abs(ySymLog(erri266[1][i],C)-NLi266[i]) for i in range(len(erri266[1]))])])
##################################################################################################################################################################################################
# data CH7_i260
i260 = np.genfromtxt('i260.txt')
num   = 34
rep   = 20
Tmeas = 30
xi260  	 = []
nli260 	 = []
for i in range(1,num+1,1):
	xi260.append(i260.T[2][(i-1)*rep:(i)*rep])
	nli260.append(NL(i260.T[0][(i-1)*rep:(i)*rep],i260.T[1][(i-1)*rep:(i)*rep],i260.T[2][(i-1)*rep:(i)*rep]))
# CH7_i260 x,xerr
Xi260     = np.mean(xi260,axis=1)/Tmeas
Xerri260  = [np.std(xi260[i]/Tmeas)/np.sqrt(len(xi260[i])) for i in range(len(xi260))]
# CH7_i260 y,yerr
meani260  = np.mean(nli260,axis=1)
stdi260   = stdNL(nli260)
#symlog recal
erri260   = [meani260 - stdi260, meani260 + stdi260]
NLi260    = [ySymLog(i,C) for i in meani260]
NLi260err = np.array([np.array([abs(ySymLog(erri260[0][i],C)-NLi260[i]) for i in range(len(erri260[0]))]),np.array([abs(ySymLog(erri260[1][i],C)-NLi260[i]) for i in range(len(erri260[1]))])])
##################################################################################################################################################################################################
# data CH7_i250
i250 = np.genfromtxt('i250.txt')
num   = 44
rep   = 20
Tmeas = 30
xi250  	 = []
nli250 	 = []
for i in range(1,num+1,1):
	xi250.append(i250.T[2][(i-1)*rep:(i)*rep])
	nli250.append(NL(i250.T[0][(i-1)*rep:(i)*rep],i250.T[1][(i-1)*rep:(i)*rep],i250.T[2][(i-1)*rep:(i)*rep]))
# CH7_i250 x,xerr
Xi250     = np.mean(xi250,axis=1)/Tmeas
Xerri250  = [np.std(xi250[i]/Tmeas)/np.sqrt(len(xi250[i])) for i in range(len(xi250))]
# CH7_i250 y,yerr
meani250  = np.mean(nli250,axis=1)
stdi250   = stdNL(nli250)
#symlog recal
erri250   = [meani250 - stdi250, meani250 + stdi250]
NLi250    = [ySymLog(i,C) for i in meani250]
NLi250err = np.array([np.array([abs(ySymLog(erri250[0][i],C)-NLi250[i]) for i in range(len(erri250[0]))]),np.array([abs(ySymLog(erri250[1][i],C)-NLi250[i]) for i in range(len(erri250[1]))])])
##################################################################################################################################################################################################
# data CH7_i200
i200 = np.genfromtxt('i200.txt')
num   = 34
rep   = 20
Tmeas = 30
xi200  	 = []
nli200 	 = []
for i in range(1,num+1,1):
	xi200.append(i200.T[2][(i-1)*rep:(i)*rep])
	nli200.append(NL(i200.T[0][(i-1)*rep:(i)*rep],i200.T[1][(i-1)*rep:(i)*rep],i200.T[2][(i-1)*rep:(i)*rep]))
# CH7_i200 x,xerr
Xi200     = np.mean(xi200,axis=1)/Tmeas
Xerri200  = [np.std(xi200[i]/Tmeas)/np.sqrt(len(xi200[i])) for i in range(len(xi200))]
# CH7_i200 y,yerr
meani200  = np.mean(nli200,axis=1)
stdi200   = stdNL(nli200)
#symlog recal
erri200   = [meani200 - stdi200, meani200 + stdi200]
NLi200    = [ySymLog(i,C) for i in meani200]
NLi200err = np.array([np.array([abs(ySymLog(erri200[0][i],C)-NLi200[i]) for i in range(len(erri200[0]))]),np.array([abs(ySymLog(erri200[1][i],C)-NLi200[i]) for i in range(len(erri200[1]))])])
##################################################################################################################################################################################################
# data CH7_i175
i175 = np.genfromtxt('i175.txt')
num   = 37
rep   = 20
Tmeas = 30
xi175  	 = []
nli175 	 = []
for i in range(1,num+1,1):
	xi175.append(i175.T[2][(i-1)*rep:(i)*rep])
	nli175.append(NL(i175.T[0][(i-1)*rep:(i)*rep],i175.T[1][(i-1)*rep:(i)*rep],i175.T[2][(i-1)*rep:(i)*rep]))
# CH7_i175 x,xerr
Xi175     = np.mean(xi175,axis=1)/Tmeas
Xerri175  = [np.std(xi175[i]/Tmeas)/np.sqrt(len(xi175[i])) for i in range(len(xi175))]
# CH7_i175 y,yerr
meani175  = np.mean(nli175,axis=1)
stdi175   = stdNL(nli175)
#symlog recal
erri175   = [meani175 - stdi175, meani175 + stdi175]
NLi175    = [ySymLog(i,C) for i in meani175]
NLi175err = np.array([np.array([abs(ySymLog(erri175[0][i],C)-NLi175[i]) for i in range(len(erri175[0]))]),np.array([abs(ySymLog(erri175[1][i],C)-NLi175[i]) for i in range(len(erri175[1]))])])
##################################################################################################################################################################################################
# data CH7_i160
i160 = np.genfromtxt('i160.txt')
num   = 34
rep   = 20
Tmeas = 30
xi160  	 = []
nli160 	 = []
for i in range(1,num+1,1):
	xi160.append(i160.T[2][(i-1)*rep:(i)*rep])
	nli160.append(NL(i160.T[0][(i-1)*rep:(i)*rep],i160.T[1][(i-1)*rep:(i)*rep],i160.T[2][(i-1)*rep:(i)*rep]))

# CH7_i160 x,xerr
Xi160     = np.mean(xi160,axis=1)/Tmeas
Xerri160  = [np.std(xi160)/Tmeas/np.sqrt(len(xi160[i])) for i in range(len(xi160))]
# CH7_i160 y,yerr
meani160  = np.mean(nli160,axis=1)
stdi160   = stdNL(nli160)
#symlog recal
erri160   = [meani160 - stdi160, meani160 + stdi160]
NLi160    = [ySymLog(i,C) for i in meani160]
NLi160err = np.array([np.array([abs(ySymLog(erri160[0][i],C)-NLi160[i]) for i in range(len(erri160[0]))]),np.array([abs(ySymLog(erri160[1][i],C)-NLi160[i]) for i in range(len(erri160[1]))])])
##################################################################################################################################################################################################
# data CH7_i145
i145 = np.genfromtxt('i145.txt')
num   = 34
rep   = 20
Tmeas = 30
xi145  	 = []
nli145 	 = []
for i in range(1,num+1,1):
	xi145.append(i145.T[2][(i-1)*rep:(i)*rep])
	nli145.append(NL(i145.T[0][(i-1)*rep:(i)*rep],i145.T[1][(i-1)*rep:(i)*rep],i145.T[2][(i-1)*rep:(i)*rep]))
# CH7_i145 x,xerr
Xi145     = np.mean(xi145,axis=1)/Tmeas
Xerri145  = [np.std(xi145[i]/Tmeas)/np.sqrt(len(xi145[i])) for i in range(len(xi145))]
# CH7_i145 y,yerr
meani145  = np.mean(nli145,axis=1)
stdi145   = stdNL(nli145)
#symlog recal
erri145   = [meani145 - stdi145, meani145 + stdi145]
NLi145    = [ySymLog(i,C) for i in meani145]
NLi145err = np.array([np.array([abs(ySymLog(erri145[0][i],C)-NLi145[i]) for i in range(len(erri145[0]))]),np.array([abs(ySymLog(erri145[1][i],C)-NLi145[i]) for i in range(len(erri145[1]))])])
##################################################################################################################################################################################################
# data CH7_i135
i135 = np.genfromtxt('i135.txt')
num   = 23
rep   = 20
Tmeas = 30
xi135  	 = []
nli135 	 = []
for i in range(1,num+1,1):
	xi135.append(i135.T[2][(i-1)*rep:(i)*rep])
	nli135.append(NL(i135.T[0][(i-1)*rep:(i)*rep],i135.T[1][(i-1)*rep:(i)*rep],i135.T[2][(i-1)*rep:(i)*rep]))
# CH7_i135 x,xerr
Xi135     = np.mean(xi135,axis=1)/Tmeas
Xerri135  = [np.std(xi135[i]/Tmeas)/np.sqrt(len(xi135[i])) for i in range(len(xi135))]
# CH7_i135 y,yerr
meani135  = np.mean(nli135,axis=1)
stdi135   = stdNL(nli135)
#symlog recal
erri135   = [meani135 - stdi135, meani135 + stdi135]
NLi135    = [ySymLog(i,C) for i in meani135]
NLi135err = np.array([np.array([abs(ySymLog(erri135[0][i],C)-NLi135[i]) for i in range(len(erri135[0]))]),np.array([abs(ySymLog(erri135[1][i],C)-NLi135[i]) for i in range(len(erri135[1]))])])
##################################################################################################################################################################################################
###PLOT
sigma=1
stdSQ=3.0
x0 = np.array([13.5,14.5,16.0,17.5,20.0,25.0,26.0,26.6])
y0 = eff.T[0]
y0err = eff.T[1]+ stdSQ

x01=x0
y01=DC.T[0]
y01err=DC.T[1]

# Reversing a list using reversed() 
def Reverse(lst): 
    return [ele for ele in reversed(lst)] 
    
x02=Reverse(x0)
y02=tau.T[0]
y02err=tau.T[1]

nl    = [NLi266,NLi260,NLi200,NLi175,NLi160,NLi145,NLi135]
nlerr = [NLi266err,NLi260err,NLi200err,NLi175err,NLi160err,NLi145err,NLi135err]
y2    = [nl[i] for i in range(len(nl))]
y2err = [nlerr[i] for i in range(len(nlerr))]
x2    = [Xi266,Xi260,Xi200,Xi175,Xi160,Xi145,Xi135]
x2err = [Xerri266,Xerri260,Xerri200,Xerri175,Xerri160,Xerri145,Xerri135]

fontsize=26
tickssize=20
yfontsize=26
#makers
markS=10
markeredgewidth=0.1
cs=5
ct=5
lw=2
des=28

tab20 = [(75, 63, 114), (236, 154, 41), (168, 32, 26), (0, 105, 146),    
             (39,71,110), (36, 130, 50), (14, 71, 73), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]   

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tab20)):    
    r, g, b = tab20[i]    
    tab20[i] = (r / 255., g / 255., b / 255.) 

fig = plt.figure(figsize=(12, 18)) 
gs = gridspec.GridSpec(9, 3) 

colors=['crimson','slateblue','dodgerblue','forestgreen','gold','black','darkorange','saddlebrown']
markers=['o','o','o','o','o','o','o','o','o']
markers=['^','o','D',"p",'v','o','s',"d"]
fill_style=['full','full','full','full','full','full','full','full','full']
ax0 = plt.subplot(gs[2:4,:])
for i in range(0,len(x0)):
	ax0.errorbar(x0[i], y0[i], yerr=sigma*y0err[i], fmt='o', marker=markers[i], markersize=markS, markeredgewidth=markeredgewidth, fillstyle=fill_style[i], color = colors[i], ecolor = colors[i], capsize=cs, capthick=ct)
ax0.errorbar(effSNSPD.T[0], effSNSPD.T[1], lw=lw, linestyle='--', markersize=markS, markeredgewidth=markeredgewidth, color = 'black', ecolor = 'black', capsize=cs, capthick=ct)
ax0.set_xlim(13,27)
ax0.set_ylim(-2.5,100)
ax0.set_xticks(np.arange(13,28,1))
ax0.set_yticks(np.arange(0,101,20))
ax0.set_xlabel(r'$I_{Bias}$ [$\mu$A]',size=fontsize)
ax0.set_ylabel(r'$\eta$ [$\%$]',size=fontsize)
ax0.grid(b=True, which='major', color='lightgray', linestyle='--')
ax0.text(10.5, 100, '(b)', fontsize=des, fontweight='normal', color='black')

colors=['black','black','black','black','black','black','black','black','black']
markers=['o','o','o','o','o','o','o','o','o']
fill_style=['full','full','full','full','full','full','full','full','full']
ax01= plt.subplot(gs[0:2, :],sharex=ax0)
for i in range(0,len(x0)):
	ax01.errorbar(x01[i], y01[i], yerr=sigma*y01err[i], fmt='o', marker=markers[i], markersize=markS, markeredgewidth=markeredgewidth, fillstyle=fill_style[i], color = colors[i+1], ecolor = colors[i+1], capsize=cs, capthick=ct)
ax01.set_yscale("log")#, nonposx='clip')
ax01.set_yticks([1,10,100,1000])
ax01.set_xlim(13,27)
ax01.set_ylim(0.3,3000)
ax01.set_ylabel('R$_{0}$ [Hz]',size=fontsize)
ax01.grid(b=True, which='major', color='lightgray', linestyle='--')
ax01.text(10.5, 2000, '(a)', fontsize=des, fontweight='normal', color='black')

colors=['dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue']
fill_style=['full','full','full','full','full','full','full','full','full']
ax02 = ax01.twinx()
for i in range(0,len(x0)):
	ax02.errorbar(x02[i], y02[i], yerr=sigma*y02err[i], fmt='o', marker=markers[i], markersize=markS, markeredgewidth=markeredgewidth, fillstyle=fill_style[i], color = colors[i+1], ecolor = colors[i+1], capsize=cs, capthick=ct)
ax02.set_xlim(13,27)
ax02.set_yticks([10,12,14,16])
ax02.set_ylim(9,17)
ax02.set_ylabel(r'$\tau$ [ns]',size=fontsize, color='dodgerblue')
ax02.tick_params('y', colors='dodgerblue')

colors=['black','saddlebrown','darkorange','gold','forestgreen','dodgerblue','slateblue','crimson']
markers=["d",'s','v',"p",'D','o','^']
fill_style=['full','full','full','full','full','full','full','full']

markS=7
markeredgewidth=0.1
cs=2
ct=2
lw=3
ax1 = plt.subplot(gs[4:, :])
for i in range(len(x2)):
	ax1.errorbar(x2[i], y2[i], lw=lw, linestyle='--', marker=markers[i], markersize=markS, markeredgewidth=markeredgewidth, fillstyle=fill_style[i], color = colors[i+1], ecolor = colors[i+1], capsize=cs, capthick=ct)

markS=12
markeredgewidth=0.1
cs=5
ct=20
lw=5
ax1.grid(b=True, which='major', color='lightgray', linestyle='--')
ax1.set_xscale("log", nonposx='clip')
ax1.axhline(linewidth=4, color='white')
ax1.axhline(y=0.04, color = 'black', lw=2)
ax1.text(12.5, -0.075, '$\parallel$', fontsize=des, fontweight='normal', color='black', rotation=-30)
ax1.text(0.4*10**8, -0.075, '$\parallel$', fontsize=des, fontweight='normal', color='black', rotation=-30)
ax1.axhline(y=-0.04, color = 'black', lw=2)

ax1.errorbar(Xi250, NLi250, yerr=NLi250err, xerr=sigma*Xerri250, fmt='o', marker='o', markersize=markS, markeredgewidth=markeredgewidth, fillstyle='full', color = colors[0], ecolor = colors[0], capsize=cs, capthick=ct, elinewidth=2)

ax1.set_xlabel(r'$R_{AB}^{det}$ [Hz]',size=fontsize)
ax1.set_ylabel(r'$\Delta$',size=fontsize)
ax1.set_yticks(ySymLog_ticks)
ax1.set_yticklabels(y_tick, fontsize=yfontsize)
ax1.set_xticks(x_tick)
ax1.set_xticklabels(x_ticklabel, fontsize=yfontsize)
ax1.text(0.575, 6.15, '(c)', fontsize=des, fontweight='normal', color='black')
ax1.set_xlim(10,10**8)
ax1.set_ylim(-4,5.7)


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.3)
plt.savefig('fig4.pdf',dpi=300,bbox_inches='tight',pad_inches = 0.01)
plt.show()
