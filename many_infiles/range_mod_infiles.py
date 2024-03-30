import moordyn
import MD_file_helpers as helpers # TODO: simpler to just use MoorPy?
import numpy as np
import os

# functions
def sin (period = 150, dt = 0.001, A = 10, axis = 0, dof = 6, x_initial = np.array([0, 0, 0, 0, 0, 0]), vector_size = 6, time = np.array([0])):

    # axis 0 -> x, 1 -> y, 3 -> z
    xp = np.zeros((len(time),vector_size))
    x = np.zeros((len(time),vector_size))
    xd = np.zeros((len(time),vector_size))
    
    # wave properties
    T = period / dt
    omega = (2*np.pi)/T
    
    for i in range(len(time)):
        xp[i,axis] = A * np.sin(i*omega)

    xdp = np.zeros((len(time),vector_size))
    xold = np.zeros(vector_size)
    # calculate velocities using finite difference
    for i in range(len(time)):
        xdp [i] = (xp[i] - xold)/dt
        xold =  xp[i]
    for i in range(len(time)):
        if i == 0:
            x[i,:] = x_initial
        else:
            j = 0
            while j < vector_size:
                x[i,j:j+dof] = x[i-1,j:j+dof] + xdp[i, 0:dof] * dt # assign the sin wave to all coupled objects at the specified object
                xd[i,j:j+dof] = xdp[i, 0:dof]
                j += dof

    return x, xd

if __name__ == "__main__":

    # initalize parameters (changes in NES mass, bungee EA, rod length, etc.)

    ### MD runtime parameters
    tMax = 4 # max time for running time sereies 
    dtC = 0.001 # coupling timestep 
    time = np.arange(0, tMax, dtC) # time series
    dof = 6 # dof of one coupled object 
    num_coupled = 2 # number of coupled objects with dof above

    ### parent MD file parameters
    path = 'MooringTest/'
    root = 'vertical_double_clearance_3nes'
    extension = '.txt'
    depth = 50 # water depth
    rho = 1025 # water density

    # load in original moordyn file, modify parameters, unload file
    parent_file = helpers.System(file = path+root+extension, dirname = path, rootname = root, depth = depth, rho = rho, Fortran = False, qs = 0)

    test_files = []
    ### modify parameters
    ##### line type dictionary strings are: 'EA', 'BA', 'EI', 'Cd', 'Ca', 'CdAx', 'CaAx', 'material' (where material returns the LineType name from the infile)
    parent_file.lineTypes['bungee']['EA'] = 2.561 * 10**8 
    parent_file.pointList[2-1].m = 4.001 # integer 2 indicated point 2
    parent_file.pointList[3-1].m = 4.001
    parent_file.pointList[4-1].m = 4.001
    parent_file.rodList[3-1].rA = np.array([0,     0.000,   -10.0009]) # integer 3 indicates rod 3
    parent_file.rodList[3-1].rB = np.array([0,     0.000,   -10.9991])
    parent_file.rodList[4-1].rA = np.array([0,     0.000,   -13.0009])  
    parent_file.rodList[4-1].rB = np.array([0,     0.000,   -13.9991])
    parent_file.rodList[5-1].rA = np.array([0,     0.000,   -17.0009])  
    parent_file.rodList[5-1].rB = np.array([0,     0.000,   -17.9991])
    parent_file.rodList[8-1].rA = np.array([10,     0.000,   -10.0009])  
    parent_file.rodList[8-1].rB = np.array([10,     0.000,   -10.9991])
    parent_file.rodList[9-1].rA = np.array([10,     0.000,   -13.0009])  
    parent_file.rodList[9-1].rB = np.array([10,     0.000,   -13.9991])
    parent_file.rodList[10-1].rA = np.array([10,     0.000,   -17.0009])  
    parent_file.rodList[10-1].rB = np.array([10,     0.000,   -17.9991])
    # parent_file.lineList[10].lUnstr = <new unstr len with different rod size> # integer 10 indicates line 10
    
    # unload to a range of files
    new_name = path+'test_mod.txt' # new file name to be written out to
    parent_file.unload(fileName=new_name, MDversion=2)
    test_files.append(new_name)
    test_files.append(path+root+extension)
    exit()

    # run MD for all the files (one loop, multiple systems running in simultaneously)
    
    ### initalize MD 
    vector_size = dof * num_coupled # 6dof coupled object * num coupled

    size = (len(time), vector_size) 
    x = np.zeros(size) 
    xd = np.zeros(size)

    x, xd = sin(period = 5, dt = dtC, A = 1, axis = 0, dof = dof, x_initial = np.array([0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0]), vector_size = vector_size, time = time) # sin excitation of connections

    systems = [] # list of MD instances

    # initalize all the MD systems
    for j, file in enumerate(test_files):
        system = moordyn.Create(file) # need to check directories work here
        systems.append(system)
        moordyn.Init(systems[j], x[0,:], xd[0,:])

    # loop through coupling time steps
    print("MoorDyn systems initialized - now performing calls to MoorDynStep...")
    for i in range(1, len(time)):
        # call the MoorDyn step function for each system
        for j in range(len(test_files)):        
            moordyn.Step(systems[j], x[i,:], xd[i,:], time[i], dtC)    #force value returned here in array

    print("Successfuly simulated for {} seconds - now closing MoorDyn...".format(tMax))  

    # close MoorDyn simulations when finished
    for j in range(len(test_files)):
        moordyn.Close(systems[j])   