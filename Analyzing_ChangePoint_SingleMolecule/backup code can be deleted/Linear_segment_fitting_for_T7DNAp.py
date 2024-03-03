# These codes were writen in python3 to calculate the T7 DNAP replication rate, which is adapted based on a linear segment fitting method.
# Before running these codes, you might need to pip install some modules if you didn't have installed with python;
# and check the file path to be exact the one you want to analyze.

from __future__ import division
from sympy import *
from sympy import coth
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nptdms import TdmsFile
import pwlf
from more_itertools import chunked

# read raw .tdms file with TdmsFile function
name = input('please type in the file name:')
tdms_filename = 'Your file path (e.g.C:\\Users\\86183\\Desktop\\)' + name + '.tdms'
tdms_file = TdmsFile(tdms_filename)

force_channel = tdms_file.object('FD Data', 'Force Channel 0 (pN)')
force = force_channel.data
time_channel = tdms_file.object('FD Data', 'Time (ms)')
time = time_channel.data
distance_channel = tdms_file.object('FD Data', 'Distance 1 (um)')
distance = distance_channel.data
bead_size = 1.76
distance = distance - bead_size

# select your interesting force cycle (50pN-10pN) by a time range
plt.plot(time,force)
plt.xlabel('time/ms')
plt.ylabel('Force/pN')
plt.show()

# please enter cycle number of interest
cycle = input('please type in cycle number of interest(e.g.:01):')
time_from = float(input('please type in the starting time of interest:'))
time_to = float(input('please type in the ending time of interest:'))

# here you will select and analyze your interesting cycle with corresponding time range
indtemp = np.where(time >= time_from)
time_range = time[indtemp]
distance_range = distance[indtemp]
force_range = force[indtemp]
indtemp_2 = np.where(time_range <= time_to)
time_range = time_range[indtemp_2]
distance_range = distance_range[indtemp_2]
force_range = force_range[indtemp_2]

plt.plot(time_range,distance_range)
plt.grid()
plt.xlabel('time/ms')
plt.ylabel('distance/um')
plt.show()

# parameters for tWLC model: Peter Gross, et al. Nature Physics volume 7, pages731–736(2011)
# dsDNA contour length Lc = 2.85056um; persistent length Lp = 56nm
# the twist rigidity C=440 pN nm2;
# the stretching modulus S=1500 pN;
# the twist–stretch coupling g(F) is given by: g(F) =g0+g1F,where g0=−637 pN nm, g1=17 nm
EEDds,Lc,F,Lp,C,g0,g1,S = symbols('EEDds Lc F Lp C g0 g1 S', real=True)
C = 440
g0= -637
g1 = 17
Lc = 2.85056
Lp = 56
S = 1500
# tWLC model expression:
def tWLC(F):
    EEDds = Lc*(1-0.5*(4.1/(F*Lp))**0.5 + C*F/(-(g0+g1*F)**2 + S*C))
    return (EEDds)

# parameters for FJC model: Smith, S. B., et al. Science 271, 795–799 (1996).
# ssDNA contour length Lss = 4.69504um,
# Kuhn length b = 1.5nm (persistent length is 0.75nm),
# the stretching modulus S=800pN
EEDss,Lss,b,Sss = symbols('EEDss Lss b Sss', real=True)
Lss = 4.69504
b = 1.5
Sss = 800
# FJC model expression:
def FJC(F):
    EEDss = []
    for Fext in force:
        x  = Lss * (coth(Fext * b / 4.1) - 4.1 / (Fext * b)) * (1 + Fext / Sss)
        EEDss.append(x)
    EEDss = np.array(EEDss)
    return (EEDss)

time = time_range /1000
distance = distance_range
force = force_range

EEDds = tWLC(force)
EEDss = FJC(force)
def ssDNA_percentage(F):
    ssDNA_percentage = (distance - tWLC(F)) / (FJC(F) - tWLC(F))
    return(ssDNA_percentage)

save_overview_figure = 'Your file path (e.g.C:\\Users\\86183\\Desktop\\)' + 'overview' + name + '-cycle#'+ cycle + '.png'
plt.figure(figsize=[9, 8])
plt.subplots_adjust(hspace=1, wspace=0.5)

#plot F-d curve (221)
plt.subplot(221)
plt.plot(distance,force)
plt.xlabel('Distance/um')
plt.ylabel('Force/pN')

#plot ft/dt curve (222)
ax1 = plt.subplot(222)
color = 'tab:green'
ax1.set_xlabel('Time/s')
ax1.set_ylabel('Distance/um', color=color)
ax1.plot(time,distance, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Force/pN', color=color)
ax2.plot(time,force, color=color)
ax2.tick_params(axis='y', labelcolor=color)

#plot ft/percentage-t curve(212)
ax3 = plt.subplot(212)

color = 'tab:red'
ax3.set_xlabel('Time/s')
ax3.set_ylabel('ssDNA percentage', color=color)
ax3.plot(time,ssDNA_percentage(force), color=color)
ax3.tick_params(axis='y', labelcolor=color)

ax4 = ax3.twinx()

color = 'tab:blue'
ax4.set_ylabel('Force/pN', color=color)
ax4.plot(time,force, color=color)
ax4.tick_params(axis='y', labelcolor=color)

plt.tight_layout()
plt.savefig(save_overview_figure,dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)
plt.show()
plt.close()

ssDNA_percentage = ssDNA_percentage(force)
dsDNA_percentage = 1-ssDNA_percentage
basepairs = dsDNA_percentage * 8393



time_from = float(input('please type in the starting time of pol:'))
time_to = float(input('please type in the ending time of pol:'))
# here you will select and analyze your interesting cycle with corresponding time range
indtemp = np.where(time_range >= time_from)
time_10pN = time_range[indtemp]
distance_10pN = distance_range[indtemp]
force_10pN = force_range[indtemp]
ssDNA_percentage_10pN = ssDNA_percentage[indtemp]
dsDNA_percentage_10pN = dsDNA_percentage[indtemp]
basepairs_10pN = basepairs[indtemp]
indtemp_2 = np.where(time_10pN <= time_to)
time_10pN = time_10pN[indtemp_2]
distance_10pN = distance_10pN[indtemp_2]
force_10pN = force_10pN[indtemp_2]
ssDNA_percentage_10pN = ssDNA_percentage_10pN[indtemp_2]
dsDNA_percentage_10pN = dsDNA_percentage_10pN[indtemp_2]
basepairs_10pN = basepairs_10pN[indtemp_2]

plt.plot(time_10pN,basepairs_10pN)
plt.show()

excel_filename = 'Your file path (e.g.C:\\Users\\86183\\Desktop\\)' + name + '-cycle#'+ cycle + '.xlsx'
writer = pd.ExcelWriter(excel_filename)
data_0 = {'time_10pN':time_10pN,
        'basepairs_10pN':basepairs_10pN}
df_1 = pd.DataFrame(data_0)
df_1.to_excel(writer)
writer.save()
writer.close()
data =  pd.read_excel(excel_filename)
x = data['time_10pN']
y = data['basepairs_10pN']

segment_number = int(input('please type in segment number that you want to fit:'))
# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y)
break_point = my_pwlf.fit(segment_number)

# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat = my_pwlf.predict(xHat)
#get event number
event_number = np.arange(1,my_pwlf.n_segments+1)
# Get the slopes
my_slopes = my_pwlf.slopes
# Get the model parameters
beta = my_pwlf.beta
# calculate the standard errors associated with each beta parameter
se = my_pwlf.standard_errors()

# calculate the piecewise R^2 value
R2values = np.zeros(my_pwlf.n_segments)
event_duration = np.zeros(my_pwlf.n_segments)
fragment_length = np.zeros(my_pwlf.n_segments)
event_ssr = np.zeros(my_pwlf.n_segments)
event_sst = np.zeros(my_pwlf.n_segments)
for i in range(my_pwlf.n_segments):
    # segregate the data based on break point locations
    xmin = my_pwlf.fit_breaks[i]
    xmax = my_pwlf.fit_breaks[i+1]
    xtemp = my_pwlf.x_data
    ytemp = my_pwlf.y_data
    indtemp = np.where(xtemp >= xmin)
    xtemp = my_pwlf.x_data[indtemp]
    ytemp = my_pwlf.y_data[indtemp]
    indtemp = np.where(xtemp <= xmax)
    xtemp = xtemp[indtemp]
    ytemp = ytemp[indtemp]
    duration = xtemp[-1] - xtemp[0]
    length = ytemp[-1] - ytemp[0]

    # predict for the new data
    yhattemp = my_pwlf.predict(xtemp)
    # calcualte ssr
    e = yhattemp - ytemp
    ssr = np.dot(e, e)
    # calculate sst
    ybar = np.ones(ytemp.size) * np.mean(ytemp)
    ydiff = ytemp - ybar
    sst = np.dot(ydiff, ydiff)
    R2values[i] = 1.0 - (ssr/sst)
    event_duration[i] = duration
    fragment_length[i] = length
    event_ssr[i] = ssr
    event_sst[i] = sst

save_segment_figure = 'Your file path (e.g.C:\\Users\\86183\\Desktop\\)' + name + '-cycle#'+ cycle + '.png'
plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')
plt.grid()
plt.xlabel('Time/s')
plt.ylabel('BasePairs')
plt.savefig(save_segment_figure,dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)
plt.show()
plt.close()

# write all the data into an excel file
writer = pd.ExcelWriter(excel_filename)
data_1 = {'time':time*1000,
        'distance':distance,
        'force':force}
data_2 = {'time/s': time,
        'distance/um':distance,
        'force/pN':force,
        'ssDNA_percentage':ssDNA_percentage,
        'dsDNA_percentage': dsDNA_percentage,
        'basepairs':basepairs}
data_3 = {'time/s': time_10pN,
        'distance/um':distance_10pN,
        'force/pN':force_10pN,
        'ssDNA_percentage':ssDNA_percentage_10pN,
        'dsDNA_percentage':dsDNA_percentage_10pN,
        'basepairs':basepairs_10pN}
data_4 = {'event_number': event_number,
        'rate(um/s)':my_slopes,
        'duration(s)':event_duration,
        'fragment_length(um)':fragment_length,
        'SSR':event_ssr,
        'SST':event_sst,
        'R2values':R2values}
data_5 = {'beta':beta,
        'standard errors':se,
        'break_point':break_point}

df_1= pd.DataFrame(data_1)
df_2 = pd.DataFrame(data_2)
df_3 = pd.DataFrame(data_3)
df_4 = pd.DataFrame(data_4)
df_5 = pd.DataFrame(data_5)

df_pol_rate = pd.concat([df_4, df_5], axis=1)

df_1.to_excel(writer,sheet_name ='raw_data')
df_2.to_excel(writer,sheet_name ='exo+pol')
df_3.to_excel(writer,sheet_name ='pol')
writer.save()