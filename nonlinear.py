import numpy as np
import matplotlib.pyplot as plt

# Script for plotting the force vs displacement of the the NES implemented in MD as a rod, mass, and two bungees (lines)

# structures
class rod(): # NES rod  (rigid connection between bungee ends at line attachment) 
    def __init__ (self, n):
        self.endA_pos = np.empty((n,3))
        self.endB_pos = np.empty((n,3))

class mass(): # NES point mass
    def __init__ (self, n):
        self.pos = np.empty((n,3))
        self.force = np.empty((n,3))

class channel(): # Output channels read in from input file
    name = ''
    units = ''
    values = []

def norm(x): # computes the norm/magnitude of a vector x 
    return np.sqrt(x.dot(x))

def dot(x1, x2): # returns the dot product rounded to 4 decimal places. Avoids numerical precision error when taking arccos of dot product of two opposite vectors (rod end A and rod end B)
    return np.around(np.dot(x1,x2), decimals=4)

if __name__ == "__main__":

    # Note: this is for plotting the NES force vs relative displacment. The MD input file needs to have the following 12 channels: 
    #   Rod: Rod#NAPX/Y/Z, Rod#NBPX/Y/Z <-- 6 channels total
    #   Point: POINT#PX/Y/Z, POINT#FX/FY/FZ <-- 6 channels total

    # User specified parameters: These need to be defined by the user based on the MD output being analyzed 
    filename = 'example.out'
    rod_num = 1 # rod ID number of rod inserted along cable as NES in MD input file
    point_num = 1 # point ID number of NES mass in MD input file
    
    # Variable initalization (NO EDIT)
    channels = [] # list of channel objects with names, units, and values (list)
    t_i = 0 # time index
    l_i = 1 # line index

    # Read in the data
    f = open(filename, 'r')
        
    for line in f: # loop through each line in the file
        u_line = line.upper().strip().split()
        if (l_i == 1):
            for data in u_line:
                chan = channel()
                chan.name = data
                chan.units = ''
                chan.values = []
                channels.append(chan)
        if (l_i == 2):
            for i, data in enumerate(u_line):
                channels[i].units = data
                
        if (l_i >= 3):
            for i, data in enumerate(u_line):
                channels[i].values.append(float(data))

            t_i +=1

        l_i +=1 

    print('Read in the following channels from '+filename+':')
    for ch in channels:
        print(' '+ch.name)
    # Done reading data
    
    # Begin input processing
    ## set up objects and arrays
    NES_rod = rod(n = t_i)
    NES_mass = mass(n = t_i)
    time = np.empty(t_i)

    ## loop stuff
    time_found = False
    mass_fx_found = False
    mass_px_found = False
    mass_fy_found = False
    mass_py_found = False
    mass_fz_found = False
    mass_pz_found = False
    rod_Ax_found = False
    rod_Bx_found = False
    rod_Ay_found = False
    rod_By_found = False
    rod_Az_found = False
    rod_Bz_found = False

    ## sort channels to NES objects
    for ch in channels:
        if ('TIME' in ch.name):
            time = np.array(ch.values)
            time_found = True
        if ('POINT' in ch.name):
            if (str(point_num) in ch.name):
                if ('PX' in ch.name): 
                    NES_mass.pos[:,0] = np.array(ch.values) 
                    mass_px_found = True
                elif ('PY' in ch.name): 
                    NES_mass.pos[:,1] = np.array(ch.values)
                    mass_py_found = True
                elif ('PZ' in ch.name): 
                    NES_mass.pos[:,2] = np.array(ch.values)
                    mass_pz_found = True
                elif ('FX' in ch.name): 
                    NES_mass.force[:,0] = np.array(ch.values) 
                    mass_fx_found = True
                elif ('FY' in ch.name): 
                    NES_mass.force[:,1] = np.array(ch.values)
                    mass_fy_found = True
                elif ('FZ' in ch.name): 
                    NES_mass.force[:,2] = np.array(ch.values)
                    mass_fz_found = True
                else: print('WARNING: unrecognized output channel for point mass location or force')
            else:
                print('WARNING: Point ID num in MD output file does not match user specified point mass ID num')
        if ('ROD' in ch.name):
            if (str(rod_num) in ch.name):
                if ('A' in ch.name):
                    if ('PX' in ch.name): 
                        NES_rod.endA_pos[:,0] = np.array(ch.values) 
                        rod_Ax_found = True
                    elif ('PY' in ch.name):  
                        NES_rod.endA_pos[:,1] = np.array(ch.values)
                        rod_Ay_found = True
                    elif ('PZ' in ch.name): 
                        NES_rod.endA_pos[:,2] = np.array(ch.values)
                        rod_Az_found = True
                elif ('B' in ch.name):
                    if ('PX' in ch.name): 
                        NES_rod.endB_pos[:,0] = np.array(ch.values)
                        rod_Bx_found = True 
                    elif ('PY' in ch.name):  
                        NES_rod.endB_pos[:,1] = np.array(ch.values)
                        rod_By_found = True
                    elif ('PZ' in ch.name): 
                        NES_rod.endB_pos[:,2] = np.array(ch.values)
                        rod_Bz_found = True
                else: print('WARNING: unrecognized channel for rod end position')
            else:
                print('WARNING: Rod ID num in MD output file does not match user specified rod ID num')
    ## error checking
    if not (time_found and mass_fx_found and mass_fy_found and mass_fz_found and mass_px_found and mass_py_found and mass_pz_found and rod_Ax_found and rod_Ay_found and rod_Az_found and rod_Bx_found and rod_By_found and rod_Bz_found):
        print("ERROR: One or more required channels not found in MD output file")
        print(time_found , mass_fx_found , mass_fy_found , mass_fz_found , mass_px_found , mass_py_found , mass_pz_found , rod_Ax_found , rod_Ay_found , rod_Az_found , rod_Bx_found , rod_By_found , rod_Bz_found)
        exit()
    # End input processing

    # Process NES relative displacement and force
    
    ## Initalize arrays
    rel_disp = np.empty(t_i)
    force = np.empty(t_i)
    force_ax = np.empty(t_i)
    force_tran = np.empty(t_i)
    alpha = np.empty(t_i)

    for t in range(0,t_i):

        ### Displacement calculates distance between center of rod and mass
        
        # ##### test case definition: linear relationship @argyris can you come up with a better test case?
        # NES_mass.pos[t,:]  = [0,0,0]
        # NES_mass.force[t,:] = [10,0,0]
        # NES_rod.endA_pos[t,:] = [2,1,1]
        # NES_rod.endB_pos[t,:] = [-2,-1,-1]  
        # ##### end test case definition

        rel_rod_endA_pos = np.subtract(NES_rod.endA_pos[t,:], NES_mass.pos[t,:]) # rod end A position relative to the mass
        rel_rod_endB_pos = np.subtract(NES_rod.endB_pos[t,:], NES_mass.pos[t,:]) # rod end B position relative to the mass
        phi = np.arctan(norm(rel_rod_endA_pos) / norm(rel_rod_endB_pos)) # angle between bungees

        rel_rod_midpoint = np.add(rel_rod_endA_pos,rel_rod_endB_pos)/2 # rod midpoint relative to mass

        rel_disp[t] = norm(rel_rod_midpoint) # relative displacement between NES mass and rod center

        ### Force
        fnet = norm(NES_mass.force[t,:]) # calcaultes magnitude of force on the NES mass
        f_unit = NES_mass.force[t,:]/fnet # unit vector of force vector
        if (rel_disp[t] == 0.0):
            print("WARNING at t = ", time[t], ": NES mass located at rod center, force along displacement vector set to 0.0")
            force[t] = 0.0
        else:
            h_unit = rel_rod_midpoint / rel_disp[t] # unit vector of displacement vector
            theta_1 = np.arccos(dot(h_unit,f_unit)) # angle between force and displacement vectors (both on same coord system, origin at NES mass)
            force[t] = fnet * np.cos(theta_1) # force component on NES mass in direction of displacement vector

        ### Angle between NES mass and rod normal

        ##### Rod unit vector at the NES mass
        rel_mid_A = (np.subtract(rel_rod_endA_pos,rel_rod_midpoint)) # vector from rod midpoint to rod end A in NES mass reference frame
        rel_mid_B = (np.subtract(rel_rod_endB_pos,rel_rod_midpoint)) # vector from rod midpoint to rod end B in NES mass reference frame
        A_unit = rel_mid_A/norm(rel_mid_A) # direction of rod end A centered at NES mass
        B_unit = rel_mid_B/norm(rel_mid_B) # direction of rod end B centered at NES mass
        # N_unit = # unit normal vector from rod center

        if (abs(np.arccos(dot(A_unit,B_unit))-np.pi) > 0.001): 
            # If the angle between the rod end A and rod end B unit vectors is not within 0.001 of pi then the rod has been incorrectly translated to the NES mass location
            print("ERROR at t = ", time[t], ": Rod not correctly represented by unit vectors at NES mass")
            print("Angle between rod end A and rod end B vectors: ", np.arccos(dot(A_unit,B_unit)))
            exit()

        ##### Angle between displacement vector and rod normal (0 or 3.1415 indicates rod is NES mass is located perpendicular from rod center)
        if (rel_disp[t] == 0.0):
            print("WARNING at t = ", time[t], ": NES mass located at rod center, angle between displacement vector and rod normal set to 0.0")
            alpha[t] = 0.0
        else:
            alpha[t] = np.arccos(dot(A_unit, h_unit)) - (np.pi/2) # angle between rod unit vector and displacement vector minus pi/2 to find angle with normal

        ##### Extract NES mass force components transverse and axial to rod
        theta_2 = np.arccos(dot(A_unit,f_unit)) # angle between force and rod axial vector (both on same coord system, origin at NES mass)
        force_ax[t] = fnet * np.cos(theta_2) # force component on NES mass in axial direction of rod
        force_tran[t] = fnet * np.cos(theta_2 - (np.pi/2)) # force component on NES mass in transverse direction of rod

        if (abs(fnet - np.sqrt(force_ax[t]**2 + force_tran[t]**2)) > 0.001):
            print("ERROR at t = ", time[t], ": fnet computed from rod axial and transverse components does not match fnet from MD output")
            print("fnet_calc: ", np.sqrt(force_ax[t]**2 + force_tran[t]**2), " fnet actual: ", fnet)
            exit()

    # # Testing outputs
    # print("alpha: ",alpha[0])
    # print("force: ",force[0])
    # print("force_ax: ",force_ax[0])
    # print("force_tran: ",force_tran[0])
    
    # Plot
    fig1 = plt.figure()
    plt.plot(rel_disp,force)
    plt.ylabel('Force (N)')
    plt.xlabel('Relative Displacement (m)')
    plt.title('NES Mass force vs displacement')

    fig2 = plt.figure()
    plt.plot(time,force_ax)
    plt.ylabel('Axial Force (N)')
    plt.xlabel('Time (s)')
    plt.title('NES force in rod axial direction')

    fig3 = plt.figure()
    plt.plot(time,force_tran)
    plt.ylabel('Transverse Force (N)')
    plt.xlabel('Time (s)')
    plt.title('NES force in rod transverse direction')

    plt.show()