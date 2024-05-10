import moordyn
import numpy as np
import matplotlib.pyplot as plt



def sin(period=0.5, A=0.01, axis=0, dof=12, x_initial=np.array([0, 0, 0, 0, 0, 0]), vector_size=6,
        time=np.array([0]), active_percentage=100):
    xp = np.zeros((len(time), 12))
    T = period / dtC
    omega = (2 * np.pi) / T

    # Calculate the active time duration
    active_time_duration = tMax * (active_percentage / 100.0)#/2

    for i in range(len(time)):
        if time[i] <= active_time_duration:  # Check if within active period
            xp[i, axis] = A * np.sin(i * omega)#+A * np.sin(i * omega*2)+A * np.sin(i * omega*4)
            xp[i, axis+6] = A * np.sin(i * omega)#+A * np.sin(i * omega*2)+A * np.sin(i * omega*4)

    xdp = np.zeros((len(time), 12))
    xold = np.zeros(12)
    for i in range(len(time)):
        xdp[i] = (xp[i] - xold) / dtC
        xold = xp[i]

    x = np.zeros((len(time), vector_size))
    xd = np.zeros((len(time), vector_size))
    for i in range(len(time)):
        if i == 0:
            x[i, :] = x_initial
        else:
            j = 0
            while j < vector_size:
                x[i, j:j + dof] = x[i - 1, j:j + dof] + xdp[i, 0:dof] * dtC
                xd[i, j:j + dof] = xdp[i, 0:dof]
                j += dof

    return x, xd


# Example usage
rootname = 'vertical_no_clearance_3nes copy'
extension = '.txt'
path = ''
tMax = 8# max time for running time series
dtC = 0.00001  # coupling timestep
time = np.arange(0, tMax, dtC)  # time series
dof = 12  # size of state vector (# dof * # coupled objects)
vector_size = dof * 1  # 1 6dof coupled object

# Initialize x and xd
size = (len(time), 12)
x = np.zeros(size)
xd = np.zeros(size)

# Call the modified sin function with 50% activity period, for example
x, xd = sin(period=0.5, A=0.01, axis=0, dof=dof, x_initial=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), vector_size=vector_size,
            time=time, active_percentage=12.5)

# Creating a new figure for the displacement vs time plot
plt.figure(figsize=(12, 7))
plt.plot(time, x[:,0], label='Cable with NESs', color='blue', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (m)')
plt.show()
# Proceed with the rest of the simulation as before

# running MD
system = moordyn.Create(path+rootname+extension)
moordyn.Init(system, x[0,:], xd[0,:])
# loop through coupling time steps
print("MoorDyn initialized - now performing calls to MoorDynStep...")
for i in range(1, len(time)):
    # call the MoorDyn step function
    print(time[i])
    moordyn.Step(system, x[i,:], xd[i,:], time[i], dtC)    #force value returned here in array

print("Successfuly simulated for {} seconds - now closing MoorDyn...".format(tMax))  

# close MoorDyn simulation (clean up the internal memory, hopefully) when finished
moordyn.Close(system)   