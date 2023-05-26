#!/usr/bin/env python
# coding: utf-8

# In[25]:


#The following code is for one dataset, change file paths and repeat as needed for different chips

#needed imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#path to file directory
filepath = '/path/to/files/'

#create file path(s) to LabView out put data
example1 = filepath + '26 - Uh vs B - pGe.txt'
example2 = filepath + '27 - Uh vs B - pGe.txt'
example3 = filepath + '28 - Uh vs B - pGe.txt'

#read data files and save them to pandas dataframes
dat1 = pd.read_csv(example1, sep='\t')
dat2 = pd.read_csv(example2, sep='\t')
dat3 = pd.read_csv(example3, sep='\t')

#create list of dfs
frames = [dat1, dat2, dat3]

#use list of dfs to combine into one df
dat_pGe = pd.concat(frames)

#set Hall Voltage column to V 
#and mag field column to B
V = dat_pGe["Hall(mV)"].values
B = dat_pGe["B(kG)"].values


###########################################
#First figure

plt.figure(dpi=1200); #set dpi=1200, makes larger figure
plt.scatter(B, V, s=0.5); #plot B vs V
plt.title('Hall Voltage vs Magnetic Field - pGe'); #title
plt.ylabel('Hall Voltage [mV]'); #y axis label
plt.xlabel('Magnetic Field [kG]'); #x axis label
plt.grid(); #create grid on plot
plt.tight_layout(); #tight layout
plt.savefig('/save/file/path/fig.png') #save figure


##########################################
## Constants and aditional calculations ##
##########################################

mu0 = 4*np.pi*10**(-7) #vacuum permeability
N = 1200 #number of turns of coil
W = 10*10**-3 #width of chip
L = 17*10**-3 #length of chip
t = 1.5*10**-3 #thickness of chip
A = W*t #cross sectional area perpendicular to B
n = N/L #turn density (turns per length)
H = n*mu0*dat_pGe['MagCurr(Amp)'] #H field calc (H = field intensity)
a,b = np.polyfit(H,dat_pGe['B(kG)'],1) #linear fit of H field to B field



############################################
#Second figure

plt.figure(dpi=1200); #set dpi=1200, makes larger figure
plt.scatter(dat_pGe['MagCurr(Amp)'],dat_pGe['B(kG)'], marker=',', s=1); #plot curent vs B
plt.plot(dat_pGe['MagCurr(Amp)'], a*H+b, linewidth=2, c='orange'); #plot current vs linear fit of H
plt.legend(labels=[f'Linear Fit\n$m$ = {np.round(a, 3)}', 'Data Points']); #create legend for plots
plt.title('Current vs Magnetic Field'); #title
plt.ylabel('Magnetic Field [kG]'); #y axis label
plt.xlabel('Current [Amp]'); #x axis label
plt.grid(); #create grid on plot
plt.tight_layout(); #tight layout
plt.savefig('/save/file/path/fig.png') #save figure

