import moordyn
import numpy as np
import matplotlib.pyplot as plt
import ctypes # You might need to 'pip install ctypes'

class MD:

    def __init__ (self, dylib_path = '', x = None, xd = None, vector_size = 6):

        # store the states here
        self.x = x
        self.xd = xd
        self.vector_size = vector_size

        # -------------------- load the MoorDyn DLL ---------------------

        #Double vector pointer data type
        self.double_p = ctypes.POINTER(ctypes.c_double)

        # Make MoorDyn function prototypes and parameter lists (remember, first entry is return type, rest are args)
        MDInitProto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_double*self.vector_size), ctypes.POINTER(ctypes.c_double*self.vector_size), ctypes.c_char_p) #need to add filename option here, maybe this c_char works? #need to determine char size 
        MDStepProto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_double*self.vector_size), ctypes.POINTER(ctypes.c_double*self.vector_size), ctypes.POINTER(ctypes.c_double*self.vector_size), self.double_p, self.double_p)
        MDClosProto = ctypes.CFUNCTYPE(ctypes.c_int)

        MDInitParams = (1, "x"), (1, "xd"), (1, "infilename") 
        MDStepParams = (1, "x"), (1, "xd"), (2, "f"), (1, "t"), (1, "dtC") 

        print("MD_Driver: dylib path is ", dylib_path)
        self.MDdylib = ctypes.CDLL(dylib_path) #load moordyn dylib

        self.MDInit = MDInitProto(("MoorDynInit", self.MDdylib), MDInitParams)
        self.MDStep = MDStepProto(("MoorDynStep", self.MDdylib), MDStepParams)
        self.MDClose= MDClosProto(("MoorDynClose", self.MDdylib))  

    def Init(self, in_file):
        infile = ctypes.c_char_p(bytes(in_file, encoding='utf8'))
        # initialize MoorDyn at origin
        self.MDInit((self.x[0,:]).ctypes.data_as(ctypes.POINTER(ctypes.c_double*self.vector_size)),(self.xd[0,:]).ctypes.data_as(ctypes.POINTER(ctypes.c_double*self.vector_size)),infile)
        print("MD_Driver: MoorDyn initialized - now performing calls to MoorDynStep...")

    def Step(self, dt, time):
        dtC = ctypes.pointer(ctypes.c_double(dt))
        t = ctypes.pointer(ctypes.c_double(time))
        self.MDStep((self.x[i,:]).ctypes.data_as(ctypes.POINTER(ctypes.c_double*self.vector_size)), (self.xd[i,:]).ctypes.data_as(ctypes.POINTER(ctypes.c_double*self.vector_size)), t, dtC)

    def Close(self):
        self.MDClose()
        del self.MDdylib

def sin(period=0.5, A=0.01, axis=0, dof=6, x_initial=np.array([0, 0, 0, 0, 0, 0]), vector_size=6,
        time=np.array([0]), active_percentage=100):
    xp = np.zeros((len(time), 12))
    T = period / dtC
    omega = (2 * np.pi) / T

    # Calculate the active time duration
    active_time_duration = tMax * (active_percentage / 100.0)/2

    for i in range(len(time)):
        if time[i] <= active_time_duration:  # Check if within active period
            xp[i, axis] = A * np.sin(i * omega)+A * np.sin(i * omega*2)+A * np.sin(i * omega*4)
            xp[i, axis+6] = A * np.sin(i * omega)+A * np.sin(i * omega*2)+A * np.sin(i * omega*4)

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
rootname = 'vertical_no_clearance_3nes'
extension = '.txt'
path = ''
dylib_path = "<your path here>/MoorDynC_ryan/MoorDyn/build/source/libmoordyn.dylib" # change this to your path to the dylib
tMax = 0.31# max time for running time series
dtC = 0.00001  # coupling timestep
time = np.arange(0, tMax, dtC)  # time series
dof = 6  # size of state vector (# dof * # coupled objects)
vector_size = dof * 1  # 1 6dof coupled object

# Initialize x and xd
size = (len(time), vector_size)
x = np.zeros(size)
xd = np.zeros(size)

# Call the modified sin function with 50% activity period, for example
x, xd = sin(period=0.5, A=0.01, axis=0, dof=dof, x_initial=np.zeros(vector_size), vector_size=vector_size,
            time=time, active_percentage=12.5)

# Creating a new figure for the displacement vs time plot
plt.figure(figsize=(6, 7))
plt.plot(time, x[:,0], label='Cable with NESs', color='blue', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (m)')
plt.show()
# Proceed with the rest of the simulation as before

# running MD
system = MD(dylib_path = dylib_path, x = x, xd = xd, vector_size=vector_size)
system.Init(path+rootname+extension)
# loop through coupling time steps
print("MoorDyn initialized - now performing calls to MoorDynStep...")
for i in range(1, len(time)):
    # call the MoorDyn step function
    system.Step(dtC, time[i])    #force value returned here in array

print("Simulated for {} seconds - now closing MoorDyn...".format(tMax))  

# close MoorDyn simulation (clean up the internal memory, hopefully) when finished
system.Close()   