#!/usr/bin/env python
# coding: utf-8

# In[1]:


#needed imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as sp
import os 
import glob
from matplotlib.pyplot import cm
import matplotlib.patches as patches


# In[4]:


#path to file directory
filepath = '/path/to/files/'

####################################
# For overhead lights
####################################

#read in spectrometer data for room light spectrum
oh = (filepath + 'your_data.txt')
#set to a dataframe
oh_spec = pd.read_csv(oh,sep='\t',header=None)


# In[6]:


plt.figure(dpi=1200) #set dpi=1200, makes larger figure
plt.plot(oh_spec[0],oh_spec[1],'k',linewidth=0.7) #plot wavelength [0] vs intensity [1]
plt.title('Overhead Light Spectra',pad=10,fontsize=18) #title
plt.xlabel('Wavelength [nm]',labelpad=5,fontsize=18) #x axis label
plt.ylabel('Intensity',labelpad=10,fontsize=18) #y axis label
plt.minorticks_on() #turn on minor ticks on the axis
plt.savefig(filepath + 'figs/fig.png') #save figure


# In[7]:


#create a lorentzian needed to fit to data, most important 
#is Full Width Half Max (FWHM), this is variable gamma
def lorentzian(x, A, x0, gamma, y0):
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + y0


####################################
# For green laser spectrum
####################################

#read in spectrometer data for green laser spectrum
gstp = (filepath + 'your_data.txt')
#set to a dataframe
g_spec = pd.read_csv(gstp,sep='\t',header=None)


# In[8]:


#initial guess based on data
p0 = [0.175, 531, 1, 0]
#find fit parameters for lorentzian by fitting to data
popt, pcov = sp.curve_fit(lorentzian, g_spec[0],g_spec[1], p0=p0)
#find error for fit parameters
perr = np.sqrt(np.diag(pcov))

x_fit = np.linspace(g_spec[0].min(), g_spec[0].max(), 1000) #create array wavelength values
y_fit = lorentzian(x_fit, *popt) #create fit lorentizan using x_fit values using fit parameters

plt.figure(dpi=1200) #set dpi=1200, makes larger figure
plt.scatter(g_spec[0],g_spec[1],c='k',marker='+',s=5,alpha=0.7) #plot wavelength [0] vs intensity [1]
plt.plot(x_fit,y_fit,'m',linewidth=1.2,alpha=0.5,label='FWHM \n $\lambda$='+str(round(popt[1],3)) +' nm \n'+ '$\delta\lambda$='+str(round(popt[2],3))+' nm') #plot fit
plt.title('Green Laser Spectra',pad=10,fontsize=18) #title
plt.xlabel('Wavelength [nm]',labelpad=2,fontsize=18) #x axis label
plt.ylabel('Intensity',labelpad=10,fontsize=18) #y axis label
plt.xlim(520,550) #change x axis limits to make plot more legible
plt.minorticks_on() #turn on minor ticks on the axis
plt.legend() #append legend to plot
plt.savefig(filepath + 'figs/fig.png') #save figure


# In[9]:


####################################
# For HeNe laser spectrum
####################################

#read in spectrometer data for HeNe laser spectrum
henestp = (filepath + 'your_data.txt')
#set to a dataframe
hene_spec = pd.read_csv(henestp,sep='\t',header=None)


# In[10]:


#initial guess based on data
p0 = [1, 632, 1, 0]
#find fit parameters for lorentzian by fitting to data
popt, pcov = sp.curve_fit(lorentzian, hene_spec[0],hene_spec[1], p0=p0)
#find error for fit parameters
perr = np.sqrt(np.diag(pcov))

x_fit = np.linspace(hene_spec[0].min(), hene_spec[0].max(), 1000) #create array wavelength values
y_fit = lorentzian(x_fit, *popt) #create fit lorentizan using x_fit values using fit parameters


plt.figure(dpi=1200) #set dpi=1200, makes larger figure
plt.scatter(hene_spec[0],hene_spec[1],c='k',marker='+',s=5,alpha=0.7) #plot wavelength [0] vs intensity [1]
plt.plot(x_fit,y_fit,'m',linewidth=1,alpha=0.5,label='FWHM \n $\lambda$='+str(round(popt[1],3)) +' nm \n'+ '$\delta\lambda$='+str(round(popt[2],3))+' nm') #plot fit
plt.title('HeNe Laser Spectra',pad=10,fontsize=18) #title
plt.xlabel('Wavelength [nm]',labelpad=2,fontsize=18) #x axis label
plt.ylabel('Intensity',labelpad=10,fontsize=18) #y axis label
plt.xlim(620,650) #change x axis limits to make plot more legible
plt.minorticks_on() #turn on minor ticks on the axis
plt.legend() #append legend to plot
plt.savefig(filepath + 'figs/fig.png') #save figure


# In[11]:


###################################
# For HeNe tube emission spectrum
###################################

#read in spectrometer data for HeNe tube emission spectrum
estp = (filepath + 'your_data.txt')
#set to a dataframe
em_spec = pd.read_csv(estp,sep='\t',header=None)


# In[12]:


#create a list of key spectral lines for expected specrtal data
wtlines = [593.9,604.6,611.8,629.4,632.8,635.2,640.1,730.5]
#create array of colors to iterate through when plotting key lines
color = cm.hsv(np.linspace(0, 1, len(wtlines)))


plt.figure(dpi=1200) #set dpi=1200, makes larger figure
plt.plot(em_spec[0],em_spec[1],'k',linewidth=1,alpha=0.5) #plot wavelength [0] vs intensity [1]

#iterate through key lines and colors to plot them to compare to data
for i,c in zip(wtlines,color):
    plt.axvline(x=i, color=c,linewidth=1,alpha=0.4,label=str(i) + ' nm')
    
plt.title('HeNe Tube Emission Spectra',pad=10,fontsize=18) #title
plt.xlabel('Wavelength [nm]',labelpad=5,fontsize=16) #x axis label
plt.ylabel('Intensity',labelpad=5,fontsize=16) #y axis label
plt.xlim(560,750) #change x axis limits to make plot more legible
plt.minorticks_on() #turn on minor ticks on the axis
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #append legend to plot
plt.subplots_adjust(right=0.7) #needed to avoid legend being cut off when fig is saved
plt.tight_layout() #apply tight layout to plot
plt.savefig(filepath + 'figs/fig.png') #save figure


# In[17]:


###################################
# For green laser picoscope data
###################################



#picoscope data is around ~40 files per, 
#the following finds all the files in the
#directory and saves their paths
path = (filepath + 'green_pico/*.txt')
all_files = glob.glob(path)

#create a list to save the dataframes for each file
dfs = []
#iterate through all picoscope files
for filename in all_files:
    df = pd.read_csv(filename, sep='\t', skiprows=[0])  # assuming tab-separated text files
    dfs.append(df) # appends df for each file to list of dfs created before

#combine all of the dfs in the list into one
g_pico = pd.concat(dfs, ignore_index=True)

#since there is so much data, aggregating it is needed to make plot legible
#group data into 0.01 ms bins and average the data in the bins
g_pico_agg = g_pico.groupby(g_pico['(ms)'] // 0.01).agg('mean')


# In[19]:


##################################
# there are two peaks per dataset
# need to fit two different
# lorentzians to these peaks
##################################

#first peak
peak1=g_pico_agg.loc[(g_pico_agg['(ms)']>2)&(g_pico_agg['(ms)']<8)] #pick two points on either side of peak to isolate
p1 = [-20, 5, 2, -38] #guess parameters from data
popt1, pcov1 = sp.curve_fit(lorentzian, peak1['(ms)'],peak1['(mV)'], p0=p1) #find fit parameters fitting model to data
perr1 = np.sqrt(np.diag(pcov1)) #find error of parameters
x1_fit = np.linspace(2, 8, 1000) #create array of wavelengths
y1_fit = lorentzian(x1_fit, *popt1) #create model output data

peak2=g_pico_agg.loc[(g_pico_agg['(ms)']>8)&(g_pico_agg['(ms)']<15)] #pick two points on either side of peak to isolate
p2 = [-10, 11, 2, -38] #guess parameters from data
popt2, pcov2 = sp.curve_fit(lorentzian, peak2['(ms)'],peak2['(mV)'], p0=p2) #find fit parameters fitting model to data
perr2 = np.sqrt(np.diag(pcov2)) #find error of parameters
x2_fit = np.linspace(8, 15, 1000) #create array of wavelengths
y2_fit = lorentzian(x2_fit, *popt2) #create model output data


#create subplots, 2 rows, 1 column, first plot will show all data and second will
#show a subset
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,5), dpi=1200, gridspec_kw={'height_ratios': [1, 2]})

#first subplot
#plot all aggregated picoscope data
axs[0].scatter(g_pico_agg['(ms)'], g_pico_agg['(mV)'], c='k', marker='+', s=1, alpha=1)
#create a rectangle to highlight subset data in plot 2
rect = plt.Rectangle((2, -39), 12, 32, linewidth=0.5, edgecolor='k', facecolor=(1,1,0,0.20))
#draw rectangle to highlight
axs[0].add_patch(rect)
#turn on minorticks
axs[0].minorticks_on()

#second subplot
#plot data
axs[1].scatter(g_pico_agg['(ms)'], g_pico_agg['(mV)'], c='k', marker='+', s=5, alpha=0.75)
#plot first peak model fit
axs[1].plot(x1_fit, y1_fit, 'r', linewidth=1.5,alpha=0.6,label='FWHM \n $x_0$='+str(round(popt1[1],2)) +' ms \n'+ '$\gamma$='+str(round(popt1[2],2))+' ms')
#plot second peak model fit
axs[1].plot(x2_fit, y2_fit, 'm', linewidth=1.5,alpha=0.6,label='$x_0$='+str(round(popt2[1],2)) +' ms \n'+ '$\gamma$='+str(round(popt2[2],2))+' ms')

axs[1].set_xlabel('Time [ms]', labelpad=5, fontsize=18) #create x label for whole plot
axs[1].minorticks_on() #turn on minior ticks
axs[1].set_xlim(0, 16) #set x axis to only show the subset of data fitted by model
axs[1].legend() #include legend

plt.suptitle('Green Laser PicoScope Data',fontsize=20) #title for whole plot 
plt.subplots_adjust(top=0.9) #adjust position of title


plt.gcf().text(0.02, 0.5, 'Voltage [mV]', va='center', rotation='vertical', fontsize=18) #create y axis label

plt.savefig(filepath + 'figs/fig.png') #save fig


# In[ ]:


###################################
# For HeNe laser picoscope data
###################################



#picoscope data is around ~40 files per, 
#the following finds all the files in the
#directory and saves their paths
path = (filepath + 'hene_pico/*.txt')
all_files = glob.glob(path)

#create a list to save the dataframes for each file
dfs = []
#iterate through all picoscope files
for filename in all_files:
    df = pd.read_csv(filename, sep='\t', skiprows=[0])  # assuming tab-separated text files
    dfs.append(df) # appends df for each file to list of dfs created before

#combine all of the dfs in the list into one
hene_pico = pd.concat(dfs, ignore_index=True)

#since there is so much data, aggregating it is needed to make plot legible
#group data into 0.01 ms bins and average the data in the bins
hene_pico_agg = hene_pico.groupby(hene_pico['(ms)'] // 0.01).agg('mean')


# In[20]:





# In[21]:





# In[24]:


##################################
# there are two peaks per dataset
# need to fit two different
# lorentzians to these peaks
##################################


#first peak
peak1=hene_pico_agg.loc[(hene_pico_agg['(ms)']>42)&(hene_pico_agg['(ms)']<45.5)] #pick two points on either side of peak to isolate
p1 = [10, 44, 2, -42] #guess parameters from data
popt1, pcov1 = sp.curve_fit(lorentzian, peak1['(ms)'],peak1['(mV)'], p0=p1) #find fit parameters fitting model to data
perr1 = np.sqrt(np.diag(pcov1)) #find error of parameters
x1_fit = np.linspace(42, 45.5, 1000) #create array of wavelengths
y1_fit = lorentzian(x1_fit, *popt1) #create model output data

#second peak
peak2=hene_pico_agg.loc[(hene_pico_agg['(ms)']>46)&(hene_pico_agg['(ms)']<49)] #pick two points on either side of peak to isolate
p2 = [10, 47, 2, -40] #guess parameters from data
popt2, pcov2 = sp.curve_fit(lorentzian, peak2['(ms)'],peak2['(mV)'], p0=p2) #find fit parameters fitting model to data
perr2 = np.sqrt(np.diag(pcov1)) #find error of parameters
x2_fit = np.linspace(46, 49, 1000) #create array of wavelengths
y2_fit = lorentzian(x2_fit, *popt2) #create model output data


# Create subplots with custom height ratio
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,5), dpi=1200, gridspec_kw={'height_ratios': [1, 2]})

#first subplot with rectangle
#plot all aggregated picoscope data
axs[0].scatter(hene_pico_agg['(ms)'], hene_pico_agg['(mV)'], c='k', marker='+', s=1, alpha=1)
#create a rectangle to highlight subset data in plot 2
rect = plt.Rectangle((42, -45), 10, 60, linewidth=0.5, edgecolor='k', facecolor=(1,1,0,0.25))
#draw rectangle to highlight
axs[0].add_patch(rect)
#set x axis to make more legible
axs[0].set_xlim(-10,70)
#turn on minor ticks
axs[0].minorticks_on()

#second subplot
#plot data
axs[1].scatter(hene_pico_agg['(ms)'], hene_pico_agg['(mV)'], c='k', marker='+', s=5, alpha=0.75)
#plot first peak model fit
axs[1].plot(x1_fit, y1_fit, 'r', linewidth=1.5,alpha=0.6, label='FWHM \n $x_0$=' + str(round(popt1[1], 2)) + ' ms \n' + '$\gamma$=' + str(round(popt1[2], 2)) + ' ms')
#plot second peak model fit
axs[1].plot(x2_fit, y2_fit, 'm', linewidth=1.5,alpha=0.6, label='$x_0$=' + str(round(popt2[1], 2)) + ' ms \n' + '$\gamma$=' + str(round(popt2[2], 2)) + ' ms')

axs[1].set_xlabel('Time [ms]', labelpad=5, fontsize=18) #create x label for whole plot
axs[1].minorticks_on() #turn on minior ticks
axs[1].set_xlim(40,50) #set x axis to only show the subset of data fitted by model
axs[1].legend() #include legend

plt.suptitle('HeNe Laser PicoScope Data', fontsize=20) #title for whole plot 
plt.subplots_adjust(top=0.9) #adjust position of title


plt.gcf().text(0.03, 0.5, 'Voltage [mV]', va='center', rotation='vertical', fontsize=18) #create y axis label

plt.savefig(filepath + 'figs/henepico.png') #save figure


# In[26]:


####################################################
# For HeNe laser single cavity mode picoscope data
####################################################

#picoscope data is around ~40 files per, 
#the following finds all the files in the
#directory and saves their paths
path = (filepath + 'hene_single_mode_pico/*.txt')
all_files = glob.glob(path)

#create a list to save the dataframes for each file
dfs = []
#iterate through all picoscope files
for filename in all_files:
    df = pd.read_csv(filename, sep='\t', skiprows=[0])  # assuming tab-separated text files
    dfs.append(df) # appends df for each file to list of dfs created before

#combine all of the dfs in the list into one
hene_mode_pico = pd.concat(dfs, ignore_index=True)

#since there is so much data, aggregating it is needed to make plot legible
#group data into 0.01 ms bins and average the data in the bins
hene_mode_pico_agg = hene_mode_pico.groupby(hene_pico['(ms)'] // 0.01).agg('mean')


# In[27]:


#fit for single mode (peak)
scav = hene_mode_pico_agg.loc[(hene_mode_pico_agg['(ms)']>40)&(hene_mode_pico_agg['(ms)']<50)] #pick two points on either side of peak to isolate
p1 = [10, 44, 2, -42] #guess parameters from data
popt1, pcov1 = sp.curve_fit(lorentzian, scav['(ms)'],scav['(mV)'], p0=p1) #find fit parameters fitting model to data
perr1 = np.sqrt(np.diag(pcov1)) #find error of parameters
x1_fit = np.linspace(43, 48, 1000) #create array of wavelengths
y1_fit = lorentzian(x1_fit, *popt1) #create model output data

# Create subplots with custom height ratio
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,5), dpi=1200, gridspec_kw={'height_ratios': [1, 2]})

#first subplot with rectangle
#plot all aggregated picoscope data
axs[0].scatter(hene_mode_pico_agg['(ms)'], hene_mode_pico_agg['(mV)'], c='k', marker='+', s=1, alpha=1)
#create a rectangle to highlight subset data in plot 2
rect = plt.Rectangle((43, -45), 5, 15, linewidth=0.5, edgecolor='k', facecolor=(1,1,0,0.25))
#draw rectangle to highlight
axs[0].add_patch(rect)
#turn on minor ticks
axs[0].minorticks_on()

#second subplot
#plot data
axs[1].scatter(hene_mode_pico_agg['(ms)'], hene_mode_pico_agg['(mV)'], c='k', marker='+', s=5, alpha=0.5)
#plot model fit
axs[1].plot(x1_fit, y1_fit, 'r', linewidth=1.5,alpha=0.8, label='FWHM \n $x_0$=' + str(round(popt1[1], 2)) + ' ms \n' + '$\gamma$=' + str(round(popt1[2], 2)) + ' ms')
#set y axis limits for legibility
axs[1].set_ylim(-45, -30)
#set x axis limits to only include subset
axs[1].set_xlim(42, 50)

axs[1].set_xlabel('Time [ms]', labelpad=5, fontsize=18) #create x label for whole plot
axs[1].minorticks_on()  #turn minor ticks on
axs[1].legend() #add legend


plt.suptitle('HeNe Single Mode PicoScope Data', fontsize=20) #title for whole plot 
plt.subplots_adjust(top=0.9)#adjust position of title


plt.gcf().text(0.02,0.5, 'Voltage [mV]', va='center', rotation='vertical', fontsize=18) #create y axis label

plt.savefig(filepath + 'figs/fig.png') #save figure

