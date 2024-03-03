# -*-coding:utf-8 -*-
import pandas as pd
import pwlf
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt

time_from = 64960
time_to = 74630
#this is the segement number you want to fit with
segment_number = 5

tdms_file = TdmsFile(r'C:\Users\86183\Desktop\20200303-181922 10pN- 30nM DNAp + NO_gp2.5 + 0.1mg_ml_BSA #030-002.tdms')
#to get all of the groups in a file
# all_groups = tdms_file.groups()
#to get all of the channels in a group.
# all_channels = tdms_file.group_channels(group)

force_channel = tdms_file.object('FD Data', 'Force Channel 0 (pN)')
force = force_channel.data
time_channel = tdms_file.object('FD Data', 'Time (ms)')
time = time_channel.data
distance_channel = tdms_file.object('FD Data', 'Distance 1 (um)')
distance = distance_channel.data

#首先运行这一步，通过这一步找到合适的时间区间，然后截取
plt.plot(time,force)
plt.show()

#根据输入的时间截取相应的distance、force和time
indtemp = np.where(time >= time_from)
time_range = time[indtemp]
distance_range = distance[indtemp]
force_range = force[indtemp]
indtemp_2 = np.where(time_range <= time_to)
time_range = time_range[indtemp_2]
distance_range = distance_range[indtemp_2]
force_range = force_range[indtemp_2]

#输出截取的time、distance和force
# print('time_range:',time_range)
# print('distance_range:',distance_range)
# print('force_range:',force_range)

x = time_range
y = distance_range

# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# fit the data for four line segments
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

# calcualte the R^2 value
# Rsquared = my_pwlf.r_squared()

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

data_1 = {'event_number': event_number,
        'rate':my_slopes,
        'duration':event_duration,
        'fragment_length':fragment_length,
        'SSR':event_ssr,
        'SST':event_sst,
        'R2values':R2values}
data_2 = {'beta':beta,
        'standard errors':se,
        'break_point':break_point}
df_1= pd.DataFrame(data_1)
df_2 = pd.DataFrame(data_2)
writer = pd.ExcelWriter(r'C:\Users\86183\Desktop\breakpoint_analyzed.xlsx')
df_1.to_excel(writer,sheet_name ='importand data')
df_2.to_excel(writer,sheet_name ='other data')
writer.save()

plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')
plt.grid()
plt.xlabel('time/ms')
plt.ylabel('distance/um')
plt.show()