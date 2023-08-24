# current paths are the exact files given by Anna due to root access structure on lab PC,
# I'll rewrite the code to use direct server paths
import traces as t
import numpy as np
import os
import matplotlib.pyplot as plt

path = "/home/shared/brain_data/input_experiment"
dir_list = os.listdir(path)
 
print("Files and directories in '", path, "' :")
 
# prints all files
print(dir_list)


print("read data")


temp  = np.load(path + '/svdTemporalComponents_corr.npy')
print(temp.shape)
spat  = np.load(path + '/svdSpatialComponents.npy')
print(spat.shape)

cam_times = np.load(path + '/cameraFrameTimes.npy')
print(cam_times.shape)
cam_times_short = cam_times[::2]
print(cam_times_short.shape)

laser_on = np.load(path + '/laserOnTimes.npy')
print(laser_on.shape)
laser_off = np.load(path + '/laserOffTimes.npy')
print(laser_off.shape)


laserX = np.load(path + '/galvoXPositions.npy')
laserY = np.load(path + '/galvoYPositions.npy')
print(laserX.shape)
print(laserY.shape)
power = np.load(path + '/laserPowers.npy')


plt.plot(cam_times_short)
plt.show()


plt.plot(laser_on[:100],'bo')
plt.plot(laser_off[:100],'ro')
plt.show()



plt.plot(laserX,laserY)
plt.show()

plt.plot(power)
plt.show()

print(laser_on[0])
print(laser_off[0])
print(cam_times_short[2150:2210])
'''
print("All index value of is: ", np.where(cam_times_short == laser_on[0])[0])
print("All index value of is: ", np.where(cam_times_short == laser_off[0]))


plt.plot(cam_times_short[:500],temp[:500,:5])
plt.show()


ss = temp.T@temp
print(ss.shape)
print(ss)

n = np.linalg.norm(temp,2,0)

plt.plot(n)
plt.show()

tempn = temp/n



plt.plot(cam_times_short[:100],tempn[:100,:5])
plt.show()



plt.plot(cam_times_short[:10000],tempn[:10000,:])
plt.show()




plt.plot(cam_times_short[:200],tempn[:200,:5])
plt.show()

plt.plot(cam_times_short[:200],temp[:200,:5])
plt.show()

ts = t.TimeSeries()'''

print(laser_off - laser_on)

diff_t = cam_times_short[1:] - cam_times_short[:-1]
dt = np.mean(diff_t)


tt = np.absolute(diff_t - dt)
print(dt)

plt.plot(tt[-5:])
plt.show()

print(tt[100:-1])

print(diff_t[:20])

# pick dt = 0.025
# intepolate data for shifted points
# 