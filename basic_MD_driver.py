import moordyn
import numpy as np

def sin (period = 150, A = 10, axis = 0, dof = 6, x_initial = np.array([0, 0, 0, 0, 0, 0]), vector_size = 6, time = np.array([0])):

    # axis 0 -> x, 1 -> y, 3 -> z
    xp = np.zeros((len(time),6))
    
    # Wave properties
    T = period / dtC
    omega = (2*np.pi)/T
    
    for i in range(len(time)):
        xp[i,axis] = A * np.sin(i*omega)

    xdp = np.zeros((len(time),6))
    xold = np.zeros(6)
    # calculate velocities using finite difference
    for i in range(len(time)):
        xdp [i] = (xp[i] - xold)/dtC
        xold =  xp[i]
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

rootname = 'vertical_base_excite'
extension = '.txt'
path = ''
tMax = 0.5 # max time for running time sereies 
dtC = 0.001 # coupling timestep 
time = np.arange(0, tMax, dtC) # time series
dof = 6 # size of state vector (# dof * # coupled objects)
vector_size = dof * 1 # 1 6dof coupled object

# currently array of 0 state vectors for every time series. Change this for base excitation
size = (len(time), 6) 
x = np.zeros(size) 
xd = np.zeros(size)
# end changes

x, xd = sin(period = 5, A = 1, axis = 0, dof = dof, x_initial = np.array([0, 0, 0, 0, 0, 0]), vector_size = vector_size, time = time)

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