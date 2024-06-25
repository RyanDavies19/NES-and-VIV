import moordyn
import numpy as np
import matplotlib.pyplot as plt

# functions
def sin (period = 150, dt = 0.001, A = 0.1, axis = 0, dof = 6, x_initial = np.array([0, 0, 0, 0, 0, 0]),
         vector_size = 6, time = np.array([0]), active_percentage=100, tMax=50, f_start=0.1, f_end=20):

    # axis 0 -> x, 1 -> y, 3 -> z
    xp = np.zeros((len(time),vector_size))

    # Calculate the active time duration
    active_time_duration = tMax * (active_percentage / 100.0)

    # Compute frequency sweep rate
    sweep_rate = (f_end - f_start) / (2 * active_time_duration)

    for i in range(len(time)):
        if time[i] <= active_time_duration:  # Check if within active period
            # Calculate current frequency
            current_freq = f_start + sweep_rate * time[i]
            omega = 2 * np.pi * current_freq

            # Calculate sinusoidal value
            xp[i, axis] = A * np.sin(omega * time[i])


    xdp = np.zeros((len(time),vector_size))
    xold = np.zeros(vector_size)
    # calculate velocities using finite difference
    for i in range(len(time)):
        xdp [i] = (xp[i] - xold)/dtC
        xold =  xp[i]

    x = np.zeros((len(time), vector_size))
    xd = np.zeros((len(time), vector_size))
    for i in range(len(time)):
        if i == 0:
            x[i,:] = x_initial
        else:
            j = 0
            while j < vector_size:
                x[i,j:j+dof] = x[i-1,j:j+dof] + xdp[i, 0:dof] * dtC
                xd[i,j:j+dof] = xdp[i, 0:dof]
                j += dof

    return x, xd

rootname = 'SDOF_Rod_1bng_9_d0.05'
extension = '.txt'
path = ''
tMax = 100# max time for running time sereies
dtC = 0.0001 # coupling timestep
time = np.arange(0, tMax, dtC) # time series
dof = 6 # size of state vector (# dof * # coupled objects)
vector_size = dof * 1 # 1 6dof coupled object

# currently array of 0 state vectors for every time series. Change this for base excitation
size = (len(time), 6) 
x = np.zeros(size) 
xd = np.zeros(size)
# end changes

x, xd = sin(period=0.5, dt=dtC, A=0.1, axis=0, dof=dof, x_initial=np.array([0, 0, 0, 0, 0, 0]), vector_size=vector_size,
            time=time, active_percentage=100, tMax=100, f_start=0.1, f_end=15)  # sin excitation of connections
#x, xd = sin(period = 5, A = 1, axis = 0, dof = dof, x_initial = np.array([0, 0, 0, 0, 0, 0]), vector_size = vector_size, time = time)

# Creating a new figure for the displacement vs time plot
plt.figure(figsize=(12, 7))
plt.plot(time, x[:,0], label='Cable with NESs', color='blue', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (m)')
plt.show()

# running MD
system = moordyn.Create(path+rootname+extension)
moordyn.Init(system, x[0, :], xd[0, :])
# loop through coupling time steps
print("MoorDyn initialized - now performing calls to MoorDynStep...")
for i in range(1, len(time)):
    # call the MoorDyn step function
    print(time[i])
    moordyn.Step(system, x[i,:], xd[i,:], time[i], dtC)    #force value returned here in array

print("Successfully simulated for {} seconds - now closing MoorDyn...".format(tMax))

# close MoorDyn simulation (clean up the internal memory, hopefully) when finished
moordyn.Close(system)   
