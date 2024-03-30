# All of these are functionsand structures from MoorPy/dev commit 4e95e4055465dd6104db1a709b25ba7133bcfe0e (3/8/24)
# https://github.com/NREL/MoorPy/commit/4e95e4055465dd6104db1a709b25ba7133bcfe0e

# most of this is unused by range_mod_infiles, but it was easier to just leave all the structure rather than paring it down

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml
import warnings
from os import path

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg.dsolve import MatrixRankWarning

import numpy as np
import time
import yaml
import os
import re

 
# base class for MoorPy exceptions
class Error(Exception):
    ''' Base class for MoorPy exceptions'''
    pass

# Catenary error class
class CatenaryError(Error):
    '''Derived error class for catenary function errors. Contains an error message.'''
    def __init__(self, message):
        self.message = message

# Line Object error class
class LineError(Error):
    '''Derived error class for Line object errors. Contains an error message and the line number with the error.'''
    def __init__(self, num, message):
        self.line_num = num
        self.message = message

# Solve error class for any solver process
class SolveError(Error):
    '''Derived error class for various solver errors. Contains an error message'''
    def __init__(self, message):
        self.message = message
    

# Generic MoorPy error
class MoorPyError(Error):
    '''Derived error class for MoorPy. Contains an error message'''
    def __init__(self, message):
        self.message = str(message)


class helpers():

    def printMat(mat):
        '''Prints a matrix to a format that is specified

        Parameters
        ----------
        mat : array
            Any matrix that is to be printed.

        Returns
        -------
        None.

        '''
        for i in range(mat.shape[0]):
            print( "\t".join(["{:+8.3e}"]*mat.shape[1]).format( *mat[i,:] ))
            
    def printVec(vec):
        '''Prints a vector to a format that is specified

        Parameters
        ----------
        vec : array
            Any vector that is to be printed.

        Returns
        -------
        None.

        '''
        print( "\t".join(["{:+9.4e}"]*len(vec)).format( *vec ))



    def unitVector(r):
        '''Returns the unit vector along the direction of input vector r.'''

        L = np.linalg.norm(r)

        return r/L



    def getInterpNums(xlist, xin, istart=0):  # should turn into function in helpers
        '''
        Paramaters
        ----------
        xlist : array
            list of x values
        xin : float
            x value to be interpolated
        istart : int
            first lower index to try
        
        Returns
        -------
        i : int
            lower index to interpolate from
        fout : float
            fraction to return   such that y* = y[i] + fout*(y[i+1]-y[i])
        '''
        
        nx = len(xlist)
    
        if xin <= xlist[0]:  #  below lowest data point
            i = 0
            fout = 0.0
    
        elif xlist[-1] <= xin:  # above highest data point
            i = nx-1
            fout = 0.0
    
        else:  # within the data range
    
            # if istart is below the actual value, start with it instead of 
            # starting at 0 to save time, but make sure it doesn't overstep the array
            if xlist[min(istart,nx)] < xin:
                i1 = istart
            else:
                i1 = 0

            for i in range(i1, nx-1):
                if xlist[i+1] > xin:
                    fout = (xin - xlist[i] )/( xlist[i+1] - xlist[i] )
                    break
        
        return i, fout
        
            

    def getH(r):
        '''function gets the alternator matrix, H, that when multiplied with a vector,
        returns the cross product of r and that vector

        Parameters
        ----------
        r : array
            the position vector that another vector is from a point of interest.

        Returns
        -------
        H : matrix
            the alternator matrix for the size-3 vector, r.

        '''
        
        H = np.array([[ 0   , r[2],-r[1]],
                    [-r[2], 0   , r[0]],
                    [ r[1],-r[0], 0   ]])
        return H
        
        
    def rotationMatrix(x3,x2,x1):
        '''Calculates a rotation matrix based on order-z,y,x instrinsic (tait-bryan?) angles, meaning
        they are about the ROTATED axes. (rotation about z-axis would be (0,0,theta) )
        
        Parameters
        ----------
        x3, x2, x1: floats
            The angles that the rotated axes are from the nonrotated axes. Normally roll,pitch,yaw respectively. [rad]

        Returns
        -------
        R : matrix
            The rotation matrix
        '''
        # initialize the sines and cosines
        s1 = np.sin(x1) 
        c1 = np.cos(x1)
        s2 = np.sin(x2) 
        c2 = np.cos(x2)
        s3 = np.sin(x3) 
        c3 = np.cos(x3)
        
        # create the rotation matrix
        R = np.array([[ c1*c2,  c1*s2*s3-c3*s1,  s1*s3+c1*c3*s2],
                    [ c2*s1,  c1*c3+s1*s2*s3,  c3*s1*s2-c1*s3],
                    [   -s2,           c2*s3,           c2*c3]])
        
        return R    


    def rotatePosition(rRelPoint, rot3):
        '''Calculates the new position of a point by applying a rotation (rotates a vector by three angles)
        
        Parameters
        ----------
        rRelPoint : array
            x,y,z coordinates of a point relative to a local frame [m]
        rot3 : array
            Three angles that describe the difference between the local frame and the global frame/ Normally roll,pitch,yaw. [rad]

        Returns
        -------
        rRel : array
            The relative rotated position of the point about the local frame [m]
        '''
        
        # get rotation matrix from three provided angles
        RotMat = helpers.rotationMatrix(rot3[0], rot3[1], rot3[2])     
        
        # find location of point in unrotated reference frame about reference point
        rRel = np.matmul(RotMat,rRelPoint)    
        
        return rRel
        

    def transformPosition(rRelPoint, r6):
        '''Calculates the position of a point based on its position relative to translated and rotated 6DOF body
        
        Parameters
        ----------
        rRelPoint : array
            x,y,z coordinates of a point relative to a local frame [m]
        r6 : array
            6DOF position vector of the origin of the local frame, in the global frame coorindates [m, rad]

        Returns
        -------
        rAbs : array
            The absolute position of the point about the global frame [m]
        '''
        # note: r6 should be in global orientation frame
        
        # absolute location = rotation of relative position + absolute position of reference point
        rAbs = helpers.rotatePosition(rRelPoint, r6[3:]) + r6[:3]
            
        return rAbs
        
        
    def translateForce3to6DOF(r, Fin):
        '''Takes in a position vector and a force vector (applied at the positon), and calculates 
        the resulting 6-DOF force and moment vector.    
        
        Parameters
        ----------
        r : array
            x,y,z coordinates at which force is acting [m]
        Fin : array
            x,y,z components of force [N]

        Returns
        -------
        Fout : array
            The resulting force and moment vector [N, Nm]
        '''
        
        # initialize output vector as same dtype as input vector (to support both real and complex inputs)
        Fout = np.zeros(6, dtype=Fin.dtype) 
        
        # set the first three elements of the output vector the same as the input vector
        Fout[:3] = Fin
        
        # set the last three elements of the output vector as the cross product of r and Fin
        Fout[3:] = np.cross(r, Fin)
        
        return Fout


    def set_plot_center(ax, x=None, y=None, z=None):
        '''Sets the center point in x and y of a 3d plot'''
        
        # adjust the center point of the figure if requested, by moving out one of the bounds
        if not x is None:
            xlims = ax.get_xlim3d()
            if   x > np.mean(xlims): ax.set_xlim([xlims[0], x + (x - xlims[0])])
            elif x < np.mean(xlims): ax.set_xlim([x - (xlims[1] - x), xlims[1]])
            
        if not y is None:
            ylims = ax.get_ylim3d()
            if   y > np.mean(ylims): ax.set_ylim([ylims[0], y + (y - ylims[0])])
            elif y < np.mean(ylims): ax.set_ylim([y - (ylims[1] - y), ylims[1]])
        
        if not z is None:
            zlims = ax.get_zlim3d()
            if   z > np.mean(zlims): ax.set_zlim([zlims[0], z + (z - zlims[0])])
            elif z < np.mean(zlims): ax.set_zlim([z - (zlims[1] - z), zlims[1]])
            
        # make sure the aspect ratio stays equal
        helpers.set_axes_equal(ax)
            
        '''    
            # set the AXIS bounds on the axis (changing these bounds can change the perspective of the matplotlib figure)
            if xbounds != None:
                ax.set_xbound(xbounds[0], xbounds[1])
                ax.autoscale(enable=False, axis='x')
            if ybounds != None:
                ax.set_ybound(ybounds[0], ybounds[1])
                ax.autoscale(enable=False, axis='y')
            if zbounds != None:
                ax.set_zbound(zbounds[0], zbounds[1])
                ax.autoscale(enable=False, axis='x')
        '''

    def set_axes_equal(ax):
        '''Sets 3D plot axes to equal scale

        Parameters
        ----------
        ax : matplotlib.pyplot axes
            the axes that are to be set equal in scale to each other.

        Returns
        -------
        None.

        '''
        
        rangex = np.diff(ax.get_xlim3d())[0]
        rangey = np.diff(ax.get_ylim3d())[0]
        rangez = np.diff(ax.get_zlim3d())[0]
        
        ax.set_box_aspect([rangex, rangey, rangez])  # note: this may require a matplotlib update
        

    def quiver_data_to_segments(X, Y, Z, u, v, w, scale=1):
        '''function to help with animation of 3d quivers'''
        
        if scale < 0.0:  # negative scale input will be treated as setting the desired RMS quiver length
            scale = -scale/np.sqrt(np.mean(u**2 + v**2 + w**2))

        segments = (X, Y, Z, X+u*scale, Y+v*scale, Z+w*scale)
        segments = np.array(segments).reshape(6,-1)
        return [[[x1, y1, z1], [x2, y2, z2]] for x1, y1, z1, x2, y2, z2 in zip(*list(segments))]
        

    def dsolve2(eval_func, X0, Ytarget=[], step_func=None, args=[], tol=0.0001, ytol=0, maxIter=20, 
            Xmin=[], Xmax=[], a_max=2.0, dX_last=[], stepfac=4, display=0, dodamping=False):
        '''
        PARAMETERS
        ----------    
        eval_func : function
            function to solve (will be passed array X, and must return array Y of same size)
        X0 : array
            initial guess of X
        Ytarget : array (optional)
            target function results (Y), assumed zero if not provided
        stp_func : function (optional)
            function use for adjusting the variables (computing dX) each step. 
            If not provided, Netwon's method with finite differencing is used.
        args : list
            A list of variables (e.g. the system object) to be passed to both the eval_func and step_func
        tol : float or array
            If scalar, the*relative* convergence tolerance (applied to step size components, dX).
            If an array, must be same size as X, and specifies an absolute convergence threshold for each variable.
        ytol: float, optional
            If specified, this is the absolute error tolerance that must be satisfied. This overrides the tol setting which otherwise works based on x values.
        Xmin, Xmax 
            Bounds. by default start bounds at infinity
        a_max
            maximum step size acceleration allowed
        dX_last
            Used if you want to dictate the initial step size/direction based on a previous attempt
        '''
        success = False
        start_time = time.time()
        # process inputs and format as arrays in case they aren't already
        
        X = np.array(np.atleast_1d(X0), dtype=np.float_)         # start off design variable
        N = len(X)
        
        Xs = np.zeros([maxIter,N]) # make arrays to store X and error results of the solve
        Es = np.zeros([maxIter,N])
        dXlist = np.zeros([maxIter,N])
        dXlist2 = np.zeros([maxIter,N])
        
        damper = 1.0   # used to add a relaxation/damping factor to reduce the step size and combat instability
        
        
        # check the target Y value input
        if len(Ytarget)==N:
            Ytarget = np.array(Ytarget, dtype=np.float_)
        elif len(Ytarget)==0:
            Ytarget = np.zeros(N, dtype=np.float_)
        else:
            raise TypeError("Ytarget must be of same length as X0")
            
        # ensure all tolerances are positive
        if ytol==0:  # if not using ytol
            if np.isscalar(tol) and tol <= 0.0:
                raise ValueError('tol value passed to dsovle2 must be positive')
            elif not np.isscalar(tol) and any(np.array(tol) <= 0):
                raise ValueError('every tol entry passed to dsovle2 must be positive')
            
        # if a step function wasn't provided, provide a default one
        if step_func==None:
            if display>1:
                print("Using default finite difference step func")
            
            def step_func(X, args, Y, oths, Ytarget, err, tols, iter, maxIter):
                ''' this now assumes tols passed in is a vector and are absolute quantities'''
                J = np.zeros([N,N])       # Initialize the Jacobian matrix that has to be a square matrix with nRows = len(X)
                
                for i in range(N):             # Newton's method: perturb each element of the X variable by a little, calculate the outputs from the
                    X2 = np.array(X)                # minimizing function, find the difference and divide by the perturbation (finding dForce/d change in design variable)
                    deltaX = stepfac*tols[i]                  # note: this function uses the tols variable that is computed in dsolve based on the tol input
                    X2[i] += deltaX
                    Y2, _, _ = eval_func(X2, args)    # here we use the provided eval_func
                    
                    J[:,i] = (Y2-Y)/deltaX             # and append that column to each respective column of the Jacobian matrix
                
                if N > 1:
                    dX = -np.matmul(np.linalg.inv(J), Y-Ytarget)   # Take this nth output from the minimizing function and divide it by the jacobian (derivative)
                else:
                    if J[0,0] == 0.0:
                        raise ValueError('dsolve2 found a zero gradient')
                        
                    dX = np.array([-(Y[0]-Ytarget[0])/J[0,0]])
                    
                    if display > 1:
                        print(f" step_func iter {iter} X={X[0]:9.2e}, error={Y[0]-Ytarget[0]:9.2e}, slope={J[0,0]:9.2e}, dX={dX[0]:9.2e}")

                return dX                              # returns dX (step to make)

        
        
        # handle bounds
        if len(Xmin)==0:
            Xmin = np.zeros(N)-np.inf
        elif len(Xmin)==N:
            Xmin = np.array(Xmin, dtype=np.float_)
        else:
            raise TypeError("Xmin must be of same length as X0")
            
        if len(Xmax)==0:
            Xmax = np.zeros(N)+np.inf
        elif len(Xmax)==N:
            Xmax = np.array(Xmax, dtype=np.float_)
        else:
            raise TypeError("Xmax must be of same length as X0")
        
        
        
        if len(dX_last)==0:
            dX_last = np.zeros(N)
        else:
            dX_last = np.array(dX_last, dtype=np.float_)

        if display>0:
            print(f"Starting dsolve iterations>>>   aiming for Y={Ytarget}")

        
        for iter in range(maxIter):

            
            # call evaluation function
            Y, oths, stop = eval_func(X, args)
            
            # compute error
            err = Y - Ytarget
            
            if display==2:
                print(f"  new iteration #{iter} with RMS error {np.linalg.norm(err):8.3e}")
            if display>2:
                print(f"  new iteration #{iter} with X={X} and Y={Y}")

            Xs[iter,:] = X
            Es[iter,:] = err

            # stop if commanded by objective function
            if stop:
                break
            
            # handle tolerances input
            if np.isscalar(tol):
                tols = tol*(np.abs(X)+tol)
            else:
                tols = np.array(tol)
            
            # check maximum iteration
            if iter==maxIter-1:
                if display>0:
                    print("Failed to find solution after "+str(iter)+" iterations, with error of "+str(err))
                    
                # looks like things didn't converge, so if N=1 do a linear fit on the last 30% of points to estimate the soln
                if N==1:
                
                    m,b = np.polyfit(Es[int(0.7*iter):iter,0], Xs[int(0.7*iter):iter,0], 1)            
                    X = np.array([b])
                    Y = np.array([0.0]) 
                    print(f"Using linaer fit to estimate solution at X={b}")
                    
                break

            #>>>> COULD ALSO HAVE AN ITERATION RESTART FUNCTION? >>> 
            #  that returns a restart boolean, as well as what values to use to restart things if true. How?
            
            else: 
                dX = step_func(X, args, Y, oths, Ytarget, err, tols, iter, maxIter)
            

            #if display>2:
            #    breakpoint()

            # Make sure we're not diverging by keeping things from reversing too much.
            # Track the previous step (dX_last) and if the current step reverses too much, stop it part way.
            # Stop it at a plane part way between the current X value and the previous X value (using golden ratio, why not).  
            
            # get the point along the previous step vector where we'll draw the bounding hyperplane (could be a line, plane, or more in higher dimensions)
            Xlim = X - 0.62*dX_last
            
            # the equation for the plane we don't want to recross is then sum(X*dX_last) = sum(Xlim*dX_last)
            if np.sum((X+dX)*dX_last) < np.sum(Xlim*dX_last):         # if we cross are going to cross it
                
                alpha = np.sum((Xlim-X)*dX_last)/np.sum(dX*dX_last)    # this is how much we need to scale down dX to land on it rather than cross it
                
                if display > 2:
                    print("  limiting oscillation with alpha="+str(alpha))
                    print(f"   dX_last was {dX_last}, dX was going to be {dX}, now it'll be {alpha*dX}")
                
                dX = alpha*dX  # scale down dX
                
            # also avoid extreme accelerations in the same direction        
            for i in range(N):
                
                # should update the following for ytol >>>
                if abs(dX_last[i]) > tols[i]:                           # only worry about accelerations if the last step was non-negligible
            
                    dX_max = a_max*dX_last[i]                           # set the maximum permissible dx in each direction based an an acceleration limit
                    
                    if dX_max == 0.0:                                   # avoid a divide-by-zero case (if dX[i] was zero to start with)
                        breakpoint()
                        dX[i] = 0.0                     
                    else:    
                        a_i = dX[i]/dX_max                              # calculate ratio of desired dx to max dx
                
                        if a_i > 1.0:
                        
                            if display > 2:
                                print(f"    limiting acceleration ({1.0/a_i:6.4f}) for axis {i}")
                                print(f"     dX_last was {dX_last}, dX was going to be {dX}")
                            
                            #dX = dX*a_max/a_i  # scale it down to the maximum value
                            dX[i] = dX[i]/a_i  # scale it down to the maximum value (treat each DOF individually)
                            
                            if display > 2:
                                print(f"     now dX will be {dX}")
            
            dXlist[iter,:] = dX
            #if iter==196:
                #breakpoint() 
            
            
            # add damping if cyclic behavior is detected at the halfway point
            if dodamping and iter == int(0.5*maxIter):
                if display > 2:   print(f"dsolve2 is at iteration {iter} (50% of maxIter)")
                        
                for j in range(2,iter-1):
                    iterc = iter - j
                    if all(np.abs(X - Xs[iterc,:]) < tols):
                        print(f"dsolve2 is going in circles detected at iteration {iter}")
                        print(f"last similar point was at iteration {iterc}")
                        damper = damper * 0.9
                        break
                        
            dX = damper*dX
                
                
            # enforce bounds
            for i in range(N):
                
                if X[i] + dX[i] < Xmin[i]:
                    dX[i] = Xmin[i] - X[i]
                    
                elif X[i] + dX[i] > Xmax[i]:
                    dX[i] = Xmax[i] - X[i]

            dXlist2[iter,:] = dX
            # check for convergence
            if (ytol==0 and all(np.abs(dX) < tols)) or (ytol > 0 and all(np.abs(err) < ytol)):
            
                if display>0:
                    print("Iteration converged after "+str(iter)+" iterations with error of "+str(err)+" and dX of "+str(dX))
                    print("Solution X is "+str(X))
                
                    #if abs(err) > 10:
                    #    breakpoint()
                    
                    if display > 0:
                        print("Total run time: {:8.2f} seconds = {:8.2f} minutes".format((time.time() - start_time),((time.time() - start_time)/60)))

                
                if any(X == Xmin) or any(X == Xmax):
                    success = False
                    print("Warning: dsolve ended on a bound.")
                else:
                    success = True
                    
                break

            dX_last = 1.0*dX # remember this current value
            
            
            X = X + dX
            
        # truncate empty parts of these arrays
        Xs      = Xs     [:iter+1]
        Es      = Es     [:iter+1]
        dXlist  = dXlist [:iter+1]
        dXlist2 = dXlist2[:iter+1]

        return X, Y, dict(iter=iter, err=err, dX=dX_last, oths=oths, Xs=Xs, Es=Es, success=success, dXlist=dXlist, dXlist2=dXlist2)


    def dsolvePlot(info):
        '''Plots dsolve or dsolve solution process based on based dict of dsolve output data'''

        import matplotlib.pyplot as plt

        n = info['Xs'].shape[1]  # number of variables

        if n < 8:
            fig, ax = plt.subplots(2*n, 1, sharex=True)
            for i in range(n):
                ax[  i].plot(info['Xs'][:info['iter']+1,i])
                ax[n+i].plot(info['Es'][:info['iter']+1,i])
            ax[-1].set_xlabel("iteration")
        else:
            fig, ax = plt.subplots(n, 2, sharex=True)
            for i in range(n):
                ax[i,0].plot(info['Xs'][:info['iter']+1,i])
                ax[i,1].plot(info['Es'][:info['iter']+1,i])
            ax[-1,0].set_xlabel("iteration, X")
            ax[-1,1].set_xlabel("iteration, Error")
        plt.show()

    def getLineProps(dnommm, material, lineProps=None, source=None, name="", rho=1025.0, g=9.81, **kwargs):
        '''Sets up a dictionary that represents a mooring line type based on the 
        specified diameter and material type. The returned dictionary can serve as
        a MoorPy line type. Data used for determining these properties is a MoorPy
        lineTypes dictionary data structure, created by loadLineProps. This data
        can be passed in via the lineProps parameter, or a new data set can be
        generated based on a YAML filename or dictionary passed in via the source 
        parameter. The lineProps dictionary should be error-checked at creation,
        so it is not error check in this function for efficiency.
            
        Parameters
        ----------
        dnommm : float
            nominal diameter [mm].
        material : string
            string identifier of the material type be used.
        lineProps : dictionary
            A MoorPy lineProps dictionary data structure containing the property scaling coefficients.
        source : dict or filename (optional)
            YAML file name or dictionary containing line property scaling coefficients
        name : any dict index (optional)
            Identifier for the line type (otherwise will be generated automatically).
        rho : float (optional)
            Water density used for computing apparent (wet) weight [kg/m^3].
        g : float (optional)
            Gravitational constant used for computing weight [m/s^2].
        '''
        
        if lineProps==None and source==None:
            raise Exception("Either lineProps or source keyword arguments must be provided")
        
        # deal with the source (is it a dictionary, or reading in a new yaml?)
        if not source==None:
            lineProps = helpers.loadLineProps(source)
            if not lineProps==None:
                print('Warning: both lineProps and source arguments were passed to getLineProps. lineProps will be ignored.')
            
        # raise an error if the material isn't in the source dictionary
        if not material in lineProps:
            raise ValueError(f'Specified mooring line material, {material}, is not in the database.')
        
        # calculate the relevant properties for this specific line type
        mat = lineProps[material]       # shorthand for the sub-dictionary of properties for the material in question    
        d = dnommm*0.001                # convert nominal diameter from mm to m      
        mass = mat['mass_d2']*d**2
        MBL  = mat[ 'MBL_0'] + mat[ 'MBL_d']*d + mat[ 'MBL_d2']*d**2 + mat[ 'MBL_d3']*d**3 
        EA   = mat[  'EA_0'] + mat[  'EA_d']*d + mat[  'EA_d2']*d**2 + mat[  'EA_d3']*d**3 + mat['EA_MBL']*MBL 
        cost =(mat['cost_0'] + mat['cost_d']*d + mat['cost_d2']*d**2 + mat['cost_d3']*d**3 
                            + mat['cost_mass']*mass + mat['cost_EA']*EA + mat['cost_MBL']*MBL)
        # add in drag and added mass coefficients if available, if not, use defaults
        if 'Cd' in mat:
            Cd   = mat['Cd']
        else:
            Cd = 1.2
        if 'Cd_ax' in mat:
            CdAx = mat['Cd_ax']
        else:
            CdAx = 0.2
        if 'Ca' in mat:
            Ca = mat['Ca']
        else:
            Ca = 1.0
        if 'Ca_ax' in mat:
            CaAx = mat['Ca_ax']
        else:
            CaAx = 0.0
            
        # internally calculate the volumetric diameter using a ratio
        d_vol = mat['dvol_dnom']*d  # [m]

        # use the volumetric diameter to calculate the apparent weight per unit length 
        w = (mass - np.pi/4*d_vol**2 *rho)*g
        
        # stiffness values for viscoelastic approach 
        EAd = mat['EAd_MBL']*MBL     # dynamic stiffness constant: Krd alpha term x MBL [N]
        EAd_Lm = mat['EAd_MBL_Lm']   # dynamic stiffness Lm slope: Krd beta term (to be multiplied by mean load) [-]
        
        # Set up a main identifier for the linetype unless one is provided
        if name=="":
            typestring = f"{material}{dnommm:.0f}"  # note: previously was type instead of material, undefined
        else:
            typestring = name
        
        notes = f"made with getLineProps"

        lineType = dict(name=typestring, d_vol=d_vol, m=mass, EA=EA, w=w,
                        MBL=MBL, EAd=EAd, EAd_Lm=EAd_Lm, input_d=d,
                        cost=cost, notes=notes, material=material, Cdn=Cd, Cdt=CdAx,Can=Ca,Cat=CaAx)
        
        lineType.update(kwargs)   # add any custom arguments provided in the call to the lineType's dictionary
            
        return lineType


    def loadLineProps(source):
        '''Loads a set of MoorPy mooring line property scaling coefficients from
        a specified YAML file or passed dictionary. Any coefficients not included
        will take a default value (zero for everything except diameter ratio, 
        which is 1). It returns a dictionary containing the complete mooring line
        property scaling coefficient set to use for any provided mooring line types.
        
        Parameters
        ----------
        source : dict or filename
            YAML file name or dictionary containing line property scaling coefficients
        
        Returns
        -------
        dictionary
            LineProps dictionary listing each supported mooring line type and 
            subdictionaries of scaling coefficients for each.
        '''

        if type(source) is dict:
            source = source
        elif source == None:
            pass
        else:
            raise Exception("loadLineProps supplied with invalid source")

        if source == None:
            pass
        elif 'lineProps' in source:
            lineProps = source['lineProps']
        else:
            raise Exception("YAML file or dictionary must have a 'lineProps' field containing the data")

        
        output = dict()  # output dictionary combining default values with loaded coefficients
        
        if source != None:
            # combine loaded coefficients and default values into dictionary that will be saved for each material
            for mat, props in lineProps.items():  
                output[mat] = {}
                output[mat]['mass_d2'  ] = helpers.getFromDict(props, 'mass_d2')  # mass must scale with d^2
                output[mat]['EA_0'     ] = helpers.getFromDict(props, 'EA_0'     , default=0.0)
                output[mat]['EA_d'     ] = helpers.getFromDict(props, 'EA_d'     , default=0.0)
                output[mat]['EA_d2'    ] = helpers.getFromDict(props, 'EA_d2'    , default=0.0)
                output[mat]['EA_d3'    ] = helpers.getFromDict(props, 'EA_d3'    , default=0.0)
                output[mat]['EA_MBL'   ] = helpers.getFromDict(props, 'EA_MBL'   , default=0.0)
                output[mat]['EAd_MBL'  ] = helpers.getFromDict(props, 'EAd_MBL'  , default=0.0)
                output[mat]['EAd_MBL_Lm']= helpers.getFromDict(props, 'EAd_MBL_Lm',default=0.0)
                output[mat]['Cd'       ] = helpers.getFromDict(props, 'Cd'       , default=0.0)
                output[mat]['CdAx'     ] = helpers.getFromDict(props, 'Cd_ax'    , default=0.0)
                output[mat]['Ca'       ] = helpers.getFromDict(props, 'Ca'       , default=0.0)
                output[mat]['CaAx'     ] = helpers.getFromDict(props, 'Ca_ax'    , default=0.0)
                
                output[mat]['MBL_0'    ] = helpers.getFromDict(props, 'MBL_0'    , default=0.0)
                output[mat]['MBL_d'    ] = helpers.getFromDict(props, 'MBL_d'    , default=0.0)
                output[mat]['MBL_d2'   ] = helpers.getFromDict(props, 'MBL_d2'   , default=0.0)
                output[mat]['MBL_d3'   ] = helpers.getFromDict(props, 'MBL_d3'   , default=0.0)
                output[mat]['dvol_dnom'] = helpers.getFromDict(props, 'dvol_dnom', default=1.0)

                # special handling if material density is provided
                if 'density' in props:
                    if 'dvol_dnom' in props:
                        raise ValueError("Only one parameter can be specified to calculate the volumetric diameter. Choose either 'dvol_dnom' or 'density'.")
                    else:
                        mass_d2 = output[mat]['mass_d2']
                        material_density = helpers.getFromDict(props, 'density')
                        output[mat]['dvol_dnom'] = np.sqrt((mass_d2/material_density)*(4/np.pi))
                
                # cost coefficients
                output[mat]['cost_0'   ] = helpers.getFromDict(props, 'cost_0'   , default=0.0)
                output[mat]['cost_d'   ] = helpers.getFromDict(props, 'cost_d'   , default=0.0)
                output[mat]['cost_d2'  ] = helpers.getFromDict(props, 'cost_d2'  , default=0.0)
                output[mat]['cost_d3'  ] = helpers.getFromDict(props, 'cost_d3'  , default=0.0)
                output[mat]['cost_mass'] = helpers.getFromDict(props, 'cost_mass', default=0.0)
                output[mat]['cost_EA'  ] = helpers.getFromDict(props, 'cost_EA'  , default=0.0)
                output[mat]['cost_MBL' ] = helpers.getFromDict(props, 'cost_MBL' , default=0.0)

        return output


    def getFromDict(dict, key, shape=0, dtype=float, default=None):
        '''
        Function to streamline getting values from design dictionary from YAML file, including error checking.

        Parameters
        ----------
        dict : dict
            the dictionary
        key : string
            the key in the dictionary
        shape : list, optional
            The desired shape of the output. If not provided, assuming scalar output. If -1, any input shape is used.
        dtype : type
            Must be a python type than can serve as a function to format the input value to the right type.
        default : number, optional
            The default value to fill in if the item isn't in the dictionary. Otherwise will raise error if the key doesn't exist.
        '''
        # in future could support nested keys   if type(key)==list: ...

        if key in dict:
            val = dict[key]                                      # get the value from the dictionary
            if shape==0:                                         # scalar input expected
                if np.isscalar(val):
                    return dtype(val)
                else:
                    raise ValueError(f"Value for key '{key}' is expected to be a scalar but instead is: {val}")
            elif shape==-1:                                      # any input shape accepted
                if np.isscalar(val):
                    return dtype(val)
                else:
                    return np.array(val, dtype=dtype)
            else:
                if np.isscalar(val):                             # if a scalar value is provided and we need to produce an array (of any shape)
                    return np.tile(dtype(val), shape)

                elif np.isscalar(shape):                         # if expecting a 1D array
                    if len(val) == shape:
                        return np.array([dtype(v) for v in val])
                    else:
                        raise ValueError(f"Value for key '{key}' is not the expected size of {shape} and is instead: {val}")

                else:                                            # must be expecting a multi-D array
                    vala = np.array(val, dtype=dtype)            # make array

                    if list(vala.shape) == shape:                      # if provided with the right shape
                        return vala
                    elif len(shape) > 2:
                        raise ValueError("Function getFromDict isn't set up for shapes larger than 2 dimensions")
                    elif vala.ndim==1 and len(vala)==shape[1]:   # if we expect an MxN array, and an array of size N is provided, tile it M times
                        return np.tile(vala, [shape[0], 1] )
                    else:
                        raise ValueError(f"Value for key '{key}' is not a compatible size for target size of {shape} and is instead: {val}")

        else:
            if default == None:
                raise ValueError(f"Key '{key}' not found in input file...")
            else:
                if shape==0 or shape==-1:
                    return default
                else:
                    return np.tile(default, shape)


    def addToDict(dict1, dict2, key1, key2, default=None):
        '''
        Function to streamline getting values from one dictionary and 
        putting them in another dictionary (potentially under a different key),
        including error checking.

        Parameters
        ----------
        dict1 : dict
            the input dictionary
        dict2 : dict
            the output dictionary
        key1 : string
            the key in the input dictionary
        key2 : string
            the key in the output dictionary
        default : number, optional
            The default value to fill in if the item isn't in the input dictionary.
            Otherwise will raise error if the key doesn't exist.
        '''
        
        if key1 in dict1:
            val = dict1[key1]  
        else:
            if default == None:
                raise ValueError(f"Key '{key1}' not found in input dictionary...")
            else:
                val = default
        
        dict2[key2] = val


    def drawBox(ax, r1, r2, color=[0,0,0,0.2]):
        '''Draw a box along the x-y-z axes between two provided corner points.'''
        
        
        ax.plot([r1[0], r2[0]], [r1[1], r1[1]], [r1[2], r1[2]], color=color) # along x
        ax.plot([r1[0], r2[0]], [r2[1], r2[1]], [r1[2], r1[2]], color=color)
        ax.plot([r1[0], r2[0]], [r1[1], r1[1]], [r2[2], r2[2]], color=color)
        ax.plot([r1[0], r2[0]], [r2[1], r2[1]], [r2[2], r2[2]], color=color)
        ax.plot([r1[0], r1[0]], [r1[1], r2[1]], [r1[2], r1[2]], color=color) # along y
        ax.plot([r2[0], r2[0]], [r1[1], r2[1]], [r1[2], r1[2]], color=color)
        ax.plot([r1[0], r1[0]], [r1[1], r2[1]], [r2[2], r2[2]], color=color)
        ax.plot([r2[0], r2[0]], [r1[1], r2[1]], [r2[2], r2[2]], color=color)
        ax.plot([r1[0], r1[0]], [r1[1], r1[1]], [r1[2], r2[2]], color=color) # along z
        ax.plot([r1[0], r1[0]], [r2[1], r2[1]], [r1[2], r2[2]], color=color)
        ax.plot([r2[0], r2[0]], [r1[1], r1[1]], [r1[2], r2[2]], color=color)
        ax.plot([r2[0], r2[0]], [r2[1], r2[1]], [r1[2], r2[2]], color=color)


    def makeTower(twrH, twrRad):
        '''Sets up mesh points for visualizing a cylindrical structure (should align with RAFT eventually.'''
        
        n = 8
        X = []
        Y = []
        Z = []
        ax=np.zeros(n+1)
        ay=np.zeros(n+1)
        for jj in range(n+1):
            ax[jj] = np.cos(float(jj)/float(n)*2.0*np.pi)
            ay[jj] = np.sin(float(jj)/float(n)*2.0*np.pi)
            
        for ii in range(int(len(twrRad)-1)):
            z0 = twrH*float(ii)/float(len(twrRad)-1)
            z1 = twrH*float(ii+1)/float(len(twrRad)-1)
            for jj in range(n+1):
                X.append(twrRad[ii]*ax[jj])
                Y.append(twrRad[ii]*ay[jj])
                Z.append(z0)            
                X.append(twrRad[ii+1]*ax[jj])
                Y.append(twrRad[ii+1]*ay[jj])
                Z.append(z1)
        
        Xs = np.array(X)
        Ys = np.array(Y)
        Zs = np.array(Z)    
        
        return Xs, Ys, Zs


    def readBathymetryFile(filename):
        '''Read a MoorDyn-style bathymetry input file (rectangular grid of depths)
        and return the lists of x and y coordinates and the matrix of depths.
        '''
        f = open(filename, 'r')

        # skip the header
        line = next(f)
        # collect the number of grid values in the x and y directions from the second and third lines
        line = next(f)
        nGridX = int(line.split()[1])
        line = next(f)
        nGridY = int(line.split()[1])
        # allocate the Xs, Ys, and main bathymetry grid arrays
        bathGrid_Xs = np.zeros(nGridX)
        bathGrid_Ys = np.zeros(nGridY)
        bathGrid = np.zeros([nGridY, nGridX])  # MH swapped order June 30
        # read in the fourth line to the Xs array
        line = next(f)
        bathGrid_Xs = [float(line.split()[i]) for i in range(nGridX)]
        # read in the remaining lines in the file into the Ys array (first entry) and the main bathymetry grid
        for i in range(nGridY):
            line = next(f)
            entries = line.split()
            bathGrid_Ys[i] = entries[0]
            bathGrid[i,:] = entries[1:]
        
        return bathGrid_Xs, bathGrid_Ys, bathGrid


    def read_mooring_file(dirName,fileName):
        # Taken from line system.... maybe should be a helper function?
        # load data from time series for single mooring line
        
        print('attempting to load '+dirName+fileName)
        
        f = open(dirName+fileName, 'r')
        
        channels = []
        units = []
        data = []
        i=0
        
        for line in f:          # loop through lines in file
        
            if (i == 0):
                for entry in line.split():      # loop over the elemets, split by whitespace
                    channels.append(entry)      # append to the last element of the list
                    
            elif (i == 1):
                for entry in line.split():      # loop over the elemets, split by whitespace
                    units.append(entry)         # append to the last element of the list
            
            elif len(line.split()) > 0:
                data.append([])  # add a new sublist to the data matrix
                import re
                r = re.compile(r"(?<=\d)\-(?=\d)")  # catch any instances where a large negative exponent has been written with the "E"
                line2 = r.sub("E-",line)            # and add in the E
                
                
                for entry in line2.split():      # loop over the elemets, split by whitespace
                    data[-1].append(entry)      # append to the last element of the list
                
            else:
                break
        
            i+=1
        
        f.close()  # close data file
        
        # use a dictionary for convenient access of channel columns (eg. data[t][ch['PtfmPitch'] )
        ch = dict(zip(channels, range(len(channels))))
        
        data2 = np.array(data)
        
        data3 = data2.astype(float)
        
        return data3, ch, channels, units    

    def read_output_file(dirName,fileName, skiplines=-1, hasunits=1, chanlim=999, dictionary=True):

        # load data from FAST output file
        # looks for channel names, then units (if hasunits==1), then data lines after first skipping [skiplines] lines.
        # skiplines == -1 signals to search for first channel names line based on starting channel "Time".
        
    #   print('attempting to load '+dirName+fileName)
        f = open(dirName+fileName, 'r')
        
        channels = []
        units = []
        data = []
        i=0
        
        for line in f:          # loop through lines in file
        
            if (skiplines == -1):               # special case signalling to search for "Time" at start of channel line
                entries = line.split()          # split elements by whitespace
                print(entries)
                if entries[0].count('Time') > 0 or entries[0].count('time') > 0:  # if we find the time keyword
                    skiplines = i
                    print("got skiplines="+str(i))
                else:
                    pass
        
            if (i < skiplines or skiplines < 0):        # if we haven't gotten to the first channel line or we're in search mode, skip
                pass
                
            elif (i == skiplines):
                for entry in line.split():      # loop over the elemets, split by whitespace
                    channels.append(entry)      # append to the last element of the list
                    
            elif (i == skiplines+1 and hasunits == 1):
                for entry in line.split():      # loop over the elemets, split by whitespace
                    if entry.count('kN') > 0 and entry.count('m') > 0:  # correct for a possible weird character
                        entry = '(kN-m)'
                        
                    units.append(entry)         # append to the last element of the list
            
            elif len(line.split()) > 0:
                data.append([])  # add a new sublist to the data matrix
                
                r = re.compile(r"(?<=\d)\-(?=\d)")  # catch any instances where a large negative exponent has been written with the "E"
                line2 = r.sub("E-",line)           # and add in the E
                
                j=0
                for entry in line2.split():      # loop over the elements, split by whitespace
                    if j > chanlim:
                        break
                    j+=1    
                    data[-1].append(entry)      # append to the last element of the list
        
            else:
                break
        
            i+=1
        
        f.close()  # close data file
        
        
        # use a dictionary for convenient access of channel columns (eg. data[t][ch['PtfmPitch'] )
        ch = dict(zip(channels, range(len(channels))))
        
        #print ch['WindVxi']

        data2 = np.array(data)
        
        data3 = data2.astype(float)
        
        if dictionary:
            dataDict = {}
            unitDict = {}
            for i in range(len(channels)):
                dataDict[channels[i]] = data3[:,i]
                unitDict[channels[i]] = units[i]
            return dataDict, unitDict
        else:
            return data3, ch, channels, units

class Line():
    '''A class for any mooring line that consists of a single material'''

    def __init__(self, mooringSys, num, L, lineType, nSegs=100, outs='p', cb=0, isRod=0, rod_attachment = '', attachments = [0,0]):
        '''Initialize Line attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the point object
        num : int
            indentifier number
        L : float
            line unstretched length [m]
        lineType : dict
            dictionary containing the coefficients needed to describe the line (could reference an entry of System.lineTypes).
        nSegs : int, optional
            number of segments to split the line into. Used in MoorPy just for plotting. The default is 100.
        outs : str, optional
            MoorDyn ouput flags for this line
        rod_attachment : str, optional
            Attachment string for unloading MoorDyn file
        cb : float, optional
            line seabed friction coefficient (will be set negative if line is fully suspended). The default is 0.
        isRod : boolean, optional
            determines whether the line is a rod or not. The default is 0.

        Returns
        -------
        None.

        '''
        
        self.sys    = mooringSys       # store a reference to the overall mooring system (instance of System class)
        
        self.number = num
        self.isRod = isRod
            
        self.L = L              # line unstretched length (may be modified if using nonlinear elasticity) [m]
        self.L0 = L             # line reference unstretched length [m]
        self.type = lineType    # dictionary of a System.lineTypes entry
        self.cost = {}          # empty dictionary to contain cost information
       
        if not isRod:
            self.EA = self.type['EA']  # use the default stiffness value for now (may be modified if using nonlinear elasticity) [N]
        
        self.outs = outs
        self.rod_attachment = rod_attachment
        self.lr_attachment = ['','']
        self.attached     = []         # ID numbers of any Lines attached to the Rod
        self.attachedEndB = []         # specifies which end of the line is attached (1: end B, 0: end A)


        self.nNodes = int(nSegs) + 1
        self.cb = float(cb)    # friction coefficient (will automatically be set negative if line is fully suspended)
        self.sbnorm = []    # Seabed Normal Vector (to be filled with a 3x1 normal vector describing seabed orientation)
        
        self.rA = np.zeros(3) # end coordinates
        self.rB = np.zeros(3)
        self.fA = np.zeros(3) # end forces
        self.fB = np.zeros(3)
        self.TA = 0  # end tensions [N]
        self.TB = 0
        self.KA = np.zeros([3,3])  # 3D stiffness matrix of end A [N/m]
        self.KB = np.zeros([3,3])  # 3D stiffness matrix of end B
        self.KBA= np.zeros([3,3])  # 3D stiffness matrix of cross coupling between ends
        
        self.HF = 0           # fairlead horizontal force saved for next solve
        self.VF = 0           # fairlead vertical force saved for next solve
        self.info = {}        # to hold all info provided by catenary
        
        self.qs = 1  # flag indicating quasi-static analysis (1). Set to 0 for time series data
        self.show = True      # a flag that will be set to false if we don't want to show the line (e.g. if results missing)
        self.color = 'k'
        self.lw=0.5
        
        self.fCurrent = np.zeros(3)  # total current force vector on the line [N]
    
    
    def loadData(self, dirname, rootname, sep='.MD.', id=0):
        '''Loads line-specific time series data from a MoorDyn output file'''
        
        self.qs = 0 # signals time series data
        
        if self.isRod==1:
            strtype='Rod'
        elif self.isRod==0:
            strtype='Line'
            
        if id==0:
            id = self.number
        
        filename = dirname+rootname+sep+strtype+str(id)+'.out'
        
        if path.exists(filename):


        # try:
        
            # load time series data
            data, ch, channels, units = helpers.read_mooring_file("", filename) # remember number starts on 1 rather than 0

            # get time info
            if ("Time" in ch):
                self.Tdata = data[:,ch["Time"]]
                self.dt = self.Tdata[1]-self.Tdata[0]
            else:
                raise LineError("loadData: could not find Time channel for mooring line "+str(self.number))
        
            
            nT = len(self.Tdata)  # number of time steps
            
            # check for position data <<<<<<
            
            self.xp = np.zeros([nT,self.nNodes])
            self.yp = np.zeros([nT,self.nNodes])
            self.zp = np.zeros([nT,self.nNodes])
            
            
            for i in range(self.nNodes):
                self.xp[:,i] = data[:, ch['Node'+str(i)+'px']]
                self.yp[:,i] = data[:, ch['Node'+str(i)+'py']]
                self.zp[:,i] = data[:, ch['Node'+str(i)+'pz']]
            
            '''
            if self.isRod==0:
                self.Te = np.zeros([nT,self.nNodes-1])   # read in tension data if available
                if "Seg1Te" in ch:
                    for i in range(self.nNodes-1):
                        self.Te[:,i] = data[:, ch['Seg'+str(i+1)+'Te']]
                        
                self.Ku = np.zeros([nT,self.nNodes])   # read in curvature data if available
                if "Node0Ku" in ch:
                    for i in range(self.nNodes):
                        self.Ku[:,i] = data[:, ch['Node'+str(i)+'Ku']]
            else:
                # read in Rod buoyancy force data if available
                if "Node0Box" in ch:
                    self.Bx = np.zeros([nT,self.nNodes])   
                    self.By = np.zeros([nT,self.nNodes])
                    self.Bz = np.zeros([nT,self.nNodes])
                    for i in range(self.nNodes):
                        self.Bx[:,i] = data[:, ch['Node'+str(i)+'Box']]
                        self.By[:,i] = data[:, ch['Node'+str(i)+'Boy']]
                        self.Bz[:,i] = data[:, ch['Node'+str(i)+'Boz']]

            if "Node0Ux" in ch:
                self.Ux = np.zeros([nT,self.nNodes])   # read in fluid velocity data if available
                self.Uy = np.zeros([nT,self.nNodes])
                self.Uz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Ux[:,i] = data[:, ch['Node'+str(i)+'Ux']]
                    self.Uy[:,i] = data[:, ch['Node'+str(i)+'Uy']]
                    self.Uz[:,i] = data[:, ch['Node'+str(i)+'Uz']]
            
            #Read in tension data if available
            if "Seg1Ten" in ch:
                self.Ten = np.zeros([nT,self.nNodes-1])   
                for i in range(self.nNodes-1):
                    self.Ten[:,i] = data[:, ch['Seg'+str(i+1)+'Ten']]
            '''
            
            
            
            # --- Read in additional data if available ---

            # segment tension  <<< to be changed to nodal tensions in future MD versions
            if "Seg1Ten" in ch:
                self.Tendata = True
                self.Te = np.zeros([nT,self.nNodes-1])
                for i in range(self.nNodes-1):
                    self.Te[:,i] = data[:, ch['Seg'+str(i+1)+'Ten']]
            elif "Seg1Te" in ch:
                self.Tendata = True
                self.Te = np.zeros([nT,self.nNodes-1])
                for i in range(self.nNodes-1):
                    self.Te[:,i] = data[:, ch['Seg'+str(i+1)+'Te']]
            else:
                self.Tendata = False
                        
            # curvature at node
            if "Node0Ku" in ch:
                self.Kudata = True
                self.Ku = np.zeros([nT,self.nNodes])   
                for i in range(self.nNodes):
                    self.Ku[:,i] = data[:, ch['Node'+str(i)+'Ku']]
            else:
                self.Kudata = False
            
            # water velocity data 
            if "Node0Ux" in ch:  
                self.Udata = True
                self.Ux = np.zeros([nT,self.nNodes])
                self.Uy = np.zeros([nT,self.nNodes])
                self.Uz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Ux[:,i] = data[:, ch['Node'+str(i)+'Ux']]
                    self.Uy[:,i] = data[:, ch['Node'+str(i)+'Uy']]
                    self.Uz[:,i] = data[:, ch['Node'+str(i)+'Uz']]
            else:
                self.Udata = False
                
            # buoyancy force data
            if "Node0Box" in ch:  
                self.Bdata = True
                self.Bx = np.zeros([nT,self.nNodes])
                self.By = np.zeros([nT,self.nNodes])
                self.Bz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Bx[:,i] = data[:, ch['Node'+str(i)+'Box']]
                    self.By[:,i] = data[:, ch['Node'+str(i)+'Boy']]
                    self.Bz[:,i] = data[:, ch['Node'+str(i)+'Boz']]
            else:
                self.Bdata = False
                
            # hydro drag data
            if "Node0Dx" in ch: 
                self.Ddata = True
                self.Dx = np.zeros([nT,self.nNodes])   # read in fluid velocity data if available
                self.Dy = np.zeros([nT,self.nNodes])
                self.Dz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Dx[:,i] = data[:, ch['Node'+str(i)+'Dx']]
                    self.Dy[:,i] = data[:, ch['Node'+str(i)+'Dy']]
                    self.Dz[:,i] = data[:, ch['Node'+str(i)+'Dz']]
            else:
                self.Ddata = False
                
            # weight data
            if "Node0Wx" in ch: 
                self.Wdata = True
                self.Wx = np.zeros([nT,self.nNodes])   # read in fluid velocity data if available
                self.Wy = np.zeros([nT,self.nNodes])
                self.Wz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Wx[:,i] = data[:, ch['Node'+str(i)+'Wx']]
                    self.Wy[:,i] = data[:, ch['Node'+str(i)+'Wy']]
                    self.Wz[:,i] = data[:, ch['Node'+str(i)+'Wz']]
            else:
                self.Wdata = False
            
            
            
            # initialize positions (is this used?)
            self.xpi= self.xp[0,:]
            self.ypi= self.yp[0,:]
            self.zpi= self.zp[0,:]
            
            # calculate the dynamic LBot !!!!!!! doesn't work for sloped bathymetry yet !!!!!!!!!!
            for i in range(len(self.zp[0])):
                if np.max(self.zp[:,i]) > self.zp[0,0]:
                    inode = i
                    break
                else:
                    inode = i
            self.LBotDyn = (inode-1)*self.L/(self.nNodes-1)
            
            # get length (constant)
            #self.L = np.sqrt( (self.xpi[-1]-self.xpi[0])**2 + (self.ypi[-1]-self.ypi[0])**2 + (self.zpi[-1]-self.zpi[0])**2 )
            # ^^^^^^^ why are we changing the self.L value to not the unstretched length specified in MoorDyn?
            # moved this below the dynamic LBot calculation because I wanted to use the original self.L
            # >>> this is probably needed for Rods - should look into using for Rods only <<<
            
            # check for tension data <<<<<<<
            
            self.show = True
            
        else:
            self.Tdata = []
            self.show = False
            print(f"Error geting data for {'Rod' if self.isRod else 'Line'} {self.number}: {filename}")
            print("dirname: {} or rootname: {} is incorrect".format(dirname, rootname))
            
         
        # >>> this was another option for handling issues - maybe no longer needed <<<
        #except Exception as e:
        #    # don't fail if there's an issue finding data, just flag that the line shouldn't be shown/plotted
        #    print(f"Error geting data for {'Rod' if self.isRod else 'Line'} {self.number}: ")
        #    print(e)
        #    self.show = False
        
    
    def setL(self, L):
        '''Sets the line unstretched length [m], and saves it for use with
        static-dynamic stiffness adjustments. Also reverts to static 
        stiffness to avoid an undefined state of having changing the line
        length in a state with adjusted dynamic EA and L values.'''
        self.L = L
        self.L0 = L
        self.revertToStaticStiffness()


    def getTimestep(self, Time):
        '''Get the time step to use for showing time series data'''
        
        if Time < 0: 
            ts = np.int_(-Time)  # negative value indicates passing a time step index
        else:           # otherwise it's a time in s, so find closest time step
            if len(self.Tdata) > 0:
                for index, item in enumerate(self.Tdata):                
                    ts = -1
                    if item > Time:
                        ts = index
                        break
                if ts==-1:
                    raise LineError(self.number, "getTimestep: requested time likely out of range")
            else:
                raise LineError(self.number, "getTimestep: zero time steps are stored")

        return ts
        
        

    def getLineCoords(self, Time, n=0, segmentTensions=False):
        '''Gets the updated line coordinates for drawing and plotting purposes.'''
        
        if n==0: n = self.nNodes   # <<< not used!
    
        # special temporary case to draw a rod for visualization. This assumes the rod end points have already been set somehow
        if self.qs==1 and self.isRod > 0:
        
            # make points for appropriately sized cylinder
            d = self.type['d_vol']
            Xs, Ys, Zs = helpers.makeTower(self.L, np.array([d/2, d/2]))   # add in makeTower method once you start using Rods
            
            # get unit vector and orientation matrix
            k = (self.rB-self.rA)/self.L
            Rmat = np.array(helpers.rotationMatrix(0, np.arctan2(np.hypot(k[0],k[1]), k[2]), np.arctan2(k[1],k[0])))
        
            # translate and rotate into proper position for Rod
            coords = np.vstack([Xs, Ys, Zs])
            newcoords = np.matmul(Rmat,coords)
            Xs = newcoords[0,:] + self.rA[0]
            Ys = newcoords[1,:] + self.rA[1]
            Zs = newcoords[2,:] + self.rA[2]
            
            return Xs, Ys, Zs, None
        
    
        # if a quasi-static analysis, just call the catenary function to return the line coordinates
        elif self.qs==1:
            
            self.staticSolve(profiles=1) # call with flag to tell Catenary to return node info
            
            #Xs = self.rA[0] + self.info["X"]*self.cosBeta 
            #Ys = self.rA[1] + self.info["X"]*self.sinBeta 
            #Zs = self.rA[2] + self.info["Z"]
            #Ts = self.info["Te"]
            Xs = self.Xs
            Ys = self.Ys
            Zs = self.Zs
            Ts = self.Ts
            return Xs, Ys, Zs, Ts
            
        # otherwise, count on read-in time-series data
        else:

            # figure out what time step to use
            ts = self.getTimestep(Time)
            
            # drawing rods
            if self.isRod > 0:
            
                k1 = np.array([ self.xp[ts,-1]-self.xp[ts,0], self.yp[ts,-1]-self.yp[ts,0], self.zp[ts,-1]-self.zp[ts,0] ]) / self.L # unit vector
                
                k = np.array(k1) # make copy
            
                Rmat = np.array(helpers.rotationMatrix(0, np.arctan2(np.hypot(k[0],k[1]), k[2]), np.arctan2(k[1],k[0])))  # <<< should fix this up at some point, MattLib func may be wrong
                
                # make points for appropriately sized cylinder
                d = self.type['d_vol']
                Xs, Ys, Zs = helpers.makeTower(self.L, np.array([d/2, d/2]))   # add in makeTower method once you start using Rods
                
                # translate and rotate into proper position for Rod
                coords = np.vstack([Xs, Ys, Zs])
                newcoords = np.matmul(Rmat,coords)
                Xs = newcoords[0,:] + self.xp[ts,0]
                Ys = newcoords[1,:] + self.yp[ts,0]
                Zs = newcoords[2,:] + self.zp[ts,0]
                
                return Xs, Ys, Zs, None
                
            # drawing lines
            else:
                
                # handle whether or not there is tension data
                try:  # use average to go from segment tension to node tensions <<< can skip this once MD is updated to output node tensions
                    if segmentTensions:
                        Te = self.Te[ts,:]  # return tensions of segments rather than averaging to get tensions of nodes
                    else:
                        Te = 0.5*(np.append(self.Te[ts,0], self.Te[ts,:]) +np.append(self.Te[ts,:], self.Te[ts,-1]))
                except: # otherwise return zeros to avoid an error (might want a warning in some cases?)
                    Te = np.zeros(self.nNodes)
                
                return self.xp[ts,:], self.yp[ts,:], self.zp[ts,:], Te
    
    
    def getCoordinate(self, s, n=100):
        '''Returns position and tension at a specific point along the line's unstretched length'''
        
        dr =  self.rB - self.rA                 
        LH = np.hypot(dr[0], dr[1])  
            
        Ss = np.linspace(0, self.L, n)
        Xs, Ys, Zs, Ts = self.getLineCoords(0.0, n=n)
        
        X = np.interp(s, Ss, Xs)*dr[0]/LH  #?
        Y = np.interp(s, Ss, Ys)*dr[1]/LH  #?
        Z = np.interp(s, Ss, Zs)
        T = np.interp(s, Ss, Ts)
        
        # <<< is this function used for anything?  Does it make sense?
        
        return X, Y, Z, T
        
    
    
    def drawLine2d(self, Time, ax, color="k", Xuvec=[1,0,0], Yuvec=[0,0,1], Xoff=0, Yoff=0, colortension=False, cmap='rainbow', plotnodes=[], plotnodesline=[], label="", alpha=1.0):
        '''Draw the line on 2D plot (ax must be 2D)

        Parameters
        ----------
        Time : float
            time value at which to draw the line
        ax : axis
            the axis on which the line is to be drawn
        color : string, optional
            color identifier in one letter (k=black, b=blue,...). The default is "k".
        Xuvec : list, optional
            plane at which the x-axis is desired. The default is [1,0,0].
        Yuvec : list, optional
            plane at which the y-axis is desired. The default is [0,0,1].
        colortension : bool, optional
            toggle to plot the lines in a colormap based on node tensions. The default is False
        cmap : string, optional
            colormap string type to plot tensions when colortension=True. The default is 'rainbow'

        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted

        '''
        
        linebit = []  # make empty list to hold plotted lines, however many there are
        
        Xs, Ys, Zs, Ts = self.getLineCoords(Time)
        
        if self.isRod > 0:
            
            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs*Xuvec[0] + Ys*Xuvec[1] + Zs*Xuvec[2] 
            Ys2d = Xs*Yuvec[0] + Ys*Yuvec[1] + Zs*Yuvec[2] 
        
            for i in range(int(len(Xs)/2-1)):
                linebit.append(ax.plot(Xs2d[2*i:2*i+2]    ,Ys2d[2*i:2*i+2]    , lw=0.5, color=color))  # side edges
                linebit.append(ax.plot(Xs2d[[2*i,2*i+2]]  ,Ys2d[[2*i,2*i+2]]  , lw=0.5, color=color))  # end A edges
                linebit.append(ax.plot(Xs2d[[2*i+1,2*i+3]],Ys2d[[2*i+1,2*i+3]], lw=0.5, color=color))  # end B edges
        
        # drawing lines...
        else:            
            if self.qs==0:
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
            
            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs*Xuvec[0] + Ys*Xuvec[1] + Zs*Xuvec[2] + Xoff
            Ys2d = Xs*Yuvec[0] + Ys*Yuvec[1] + Zs*Yuvec[2] + Yoff
            
            if colortension:    # if the mooring lines want to be plotted with colors based on node tensions
                maxT = np.max(Ts); minT = np.min(Ts)
                for i in range(len(Xs)-1):          # for each node in the line
                    color_ratio = ((Ts[i] + Ts[i+1])/2 - minT)/(maxT - minT)  # ratio of the node tension in relation to the max and min tension
                    cmap_obj = cm.get_cmap(cmap)    # create a cmap object based on the desired colormap
                    rgba = cmap_obj(color_ratio)    # return the rbga values of the colormap of where the node tension is
                    linebit.append(ax.plot(Xs2d[i:i+2], Ys2d[i:i+2], color=rgba))
            else:
                linebit.append(ax.plot(Xs2d, Ys2d, lw=1, color=color, label=label, alpha=alpha)) # previously had lw=1 (linewidth)
            
            if len(plotnodes) > 0:
                for i,node in enumerate(plotnodes):
                    if self.number==plotnodesline[i]:
                        linebit.append(ax.plot(Xs2d[node], Ys2d[node], 'o', color=color, markersize=5))   
            
        self.linebit = linebit # can we store this internally?
        
        self.X = np.array([Xs, Ys, Zs])
            
        return linebit

    

    def drawLine(self, Time, ax, color="k", endpoints=False, shadow=True, colortension=False, cmap_tension='rainbow'):
        '''Draw the line in 3D
        
        Parameters
        ----------
        Time : float
            time value at which to draw the line
        ax : axis
            the axis on which the line is to be drawn
        color : string, optional
            color identifier in one letter (k=black, b=blue,...). The default is "k".
        endpoints : bool, optional
            toggle to plot the end points of the lines. The default is False
        shadow : bool, optional
            toggle to plot the mooring line shadow on the seabed. The default is True
        colortension : bool, optional
            toggle to plot the lines in a colormap based on node tensions. The default is False
        cmap : string, optional
            colormap string type to plot tensions when colortension=True. The default is 'rainbow'
            
        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted
        '''
        
        if not self.show:  # exit if this line isn't set to be shown
            return 0
        
        if color == 'self':
            color = self.color  # attempt to allow custom colors
            lw = self.lw
        elif color == None:
            color = [0.3, 0.3, 0.3]  # if no color, default to grey
            lw = 1
        else:
            lw = 1
        
        linebit = []  # make empty list to hold plotted lines, however many there are
        
        Xs, Ys, Zs, tensions = self.getLineCoords(Time)
        
        if self.isRod > 0:
            for i in range(int(len(Xs)/2-1)):
                linebit.append(ax.plot(Xs[2*i:2*i+2],Ys[2*i:2*i+2],Zs[2*i:2*i+2]            , color=color))  # side edges
                linebit.append(ax.plot(Xs[[2*i,2*i+2]],Ys[[2*i,2*i+2]],Zs[[2*i,2*i+2]]      , color=color))  # end A edges
                linebit.append(ax.plot(Xs[[2*i+1,2*i+3]],Ys[[2*i+1,2*i+3]],Zs[[2*i+1,2*i+3]], color=color))  # end B edges
            
            # scatter points for line ends 
            #if endpoints == True:
            #    linebit.append(ax.scatter([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], [Zs[0], Zs[-1]], color = color))
        
        # drawing lines...
        else:
            if self.qs==0:
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
            
            if colortension:    # if the mooring lines want to be plotted with colors based on node tensions
                maxT = np.max(tensions); minT = np.min(tensions)
                for i in range(len(Xs)-1):          # for each node in the line
                    color_ratio = ((tensions[i] + tensions[i+1])/2 - minT)/(maxT - minT)  # ratio of the node tension in relation to the max and min tension
                    cmap_obj = cm.get_cmap(cmap_tension)    # create a cmap object based on the desired colormap
                    rgba = cmap_obj(color_ratio)    # return the rbga values of the colormap of where the node tension is
                    linebit.append(ax.plot(Xs[i:i+2], Ys[i:i+2], Zs[i:i+2], color=rgba, zorder=100))
            else:
                linebit.append(ax.plot(Xs, Ys, Zs, color=color, lw=lw, zorder=100))
            
            if shadow:
                ax.plot(Xs, Ys, np.zeros_like(Xs)-self.sys.depth, color=[0.5, 0.5, 0.5, 0.2], lw=lw, zorder = 1.5) # draw shadow
            
            if endpoints == True:
                linebit.append(ax.scatter([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], [Zs[0], Zs[-1]], color = color))
                
                    
            # draw additional data if available (should make this for rods too eventually - drawn along their axis nodes)
            if self.qs == 0:              
                ts = self.getTimestep(Time)
                
                if self.Tendata:
                    pass
                if self.Kudata:
                    pass        
                if self.Udata:
                    self.Ubits = ax.quiver(Xs, Ys, Zs, self.Ux[ts,:], self.Uy[ts,:], self.Uz[ts,:], color="blue")  # make quiver plot and save handle to line object
                if self.Bdata:
                    self.Bbits = ax.quiver(Xs, Ys, Zs, self.Bx[ts,:], self.By[ts,:], self.Bz[ts,:], color="red")
                if self.Ddata:
                    self.Dbits = ax.quiver(Xs, Ys, Zs, self.Dx[ts,:], self.Dy[ts,:], self.Dz[ts,:], color="green")
                if self.Wdata:
                    self.Wbits = ax.quiver(Xs, Ys, Zs, self.Wx[ts,:], self.Wy[ts,:], self.Wz[ts,:], color="orange")
                
                
        self.linebit = linebit # can we store this internally?
        
        self.X = np.array([Xs, Ys, Zs])
        
            
        return linebit
    
    
    def redrawLine(self, Time, colortension=False, cmap_tension='rainbow', drawU=True):  #, linebit):
        '''Update 3D line drawing based on instantaneous position'''
        
        linebit = self.linebit
        
        if self.isRod > 0:
            
            Xs, Ys, Zs, Ts = self.getLineCoords(Time)
            
            for i in range(int(len(Xs)/2-1)):
                        
                linebit[3*i  ][0].set_data(Xs[2*i:2*i+2],Ys[2*i:2*i+2])    # side edges (x and y coordinates)
                linebit[3*i  ][0].set_3d_properties(Zs[2*i:2*i+2])         #            (z coordinates)             
                linebit[3*i+1][0].set_data(Xs[[2*i,2*i+2]],Ys[[2*i,2*i+2]])           # end A edges
                linebit[3*i+1][0].set_3d_properties(Zs[[2*i,2*i+2]])                    
                linebit[3*i+2][0].set_data(Xs[[2*i+1,2*i+3]],Ys[[2*i+1,2*i+3]])   # end B edges
                linebit[3*i+2][0].set_3d_properties(Zs[[2*i+1,2*i+3]])
        
        # drawing lines...
        else:
        
            Xs, Ys, Zs, Ts = self.getLineCoords(Time)
            
            if colortension:
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])       # update the line ends based on the MoorDyn data
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
                maxT = np.max(Ts); minT = np.min(Ts)
                cmap_obj = cm.get_cmap(cmap_tension)               # create the colormap object
                
                for i in range(len(Xs)-1):  # for each node in the line, find the relative tension of the segment based on the max and min tensions
                    color_ratio = ((Ts[i] + Ts[i+1])/2 - minT)/(maxT - minT)
                    rgba = cmap_obj(color_ratio)
                    linebit[i][0]._color = rgba         # set the color of the segment to a new color based on its updated tension
                    linebit[i][0].set_data(Xs[i:i+2],Ys[i:i+2])     # set the x and y coordinates
                    linebit[i][0].set_3d_properties(Zs[i:i+2])      # set the z coorindates
            
            else:
                linebit[0][0].set_data(Xs,Ys)    # (x and y coordinates)
                linebit[0][0].set_3d_properties(Zs)         # (z coordinates) 
                    
            
        
            # draw additional data if available (should make this for rods too eventually - drawn along their axis nodes)
            if self.qs == 0:
                ts = self.getTimestep(Time)                    
                s = 0.0002
                
                if self.Tendata:
                    pass
                if self.Kudata:
                    pass        
                if self.Udata:
                    self.Ubits.set_segments(helpers.quiver_data_to_segments(Xs, Ys, Zs, self.Ux[ts,:], self.Uy[ts,:], self.Uz[ts,:], scale=10.))
                if self.Bdata:
                    self.Bbits.set_segments(helpers.quiver_data_to_segments(Xs, Ys, Zs, self.Bx[ts,:], self.By[ts,:], self.Bz[ts,:], scale=s))
                if self.Ddata:
                    self.Dbits.set_segments(helpers.quiver_data_to_segments(Xs, Ys, Zs, self.Dx[ts,:], self.Dy[ts,:], self.Dz[ts,:], scale=s))
                if self.Wdata:
                    self.Wbits.set_segments(helpers.quiver_data_to_segments(Xs, Ys, Zs, self.Wx[ts,:], self.Wy[ts,:], self.Wz[ts,:], scale=s))
                
                
        
        return linebit
        
        
    
    
    def setEndPosition(self, r, endB):
        '''Sets the end position of the line based on the input endB value.

        Parameters
        ----------
        r : array
            x,y,z coorindate position vector of the line end [m].
        endB : boolean
            An indicator of whether the r array is at the end or beginning of the line

        Raises
        ------
        LineError
            If the given endB value is not a 1 or 0

        Returns
        -------
        None.

        '''
        
        if endB == 1:
            self.rB = np.array(r, dtype=np.float_)
        elif endB == 0:
            self.rA = np.array(r, dtype=np.float_)
        else:
            raise LineError("setEndPosition: endB value has to be either 1 or 0")
        
        
    def staticSolve(self, reset=False, tol=0.0001, profiles=0):
        '''Solves static equilibrium of line. Sets the end forces of the line based on the end points' positions.

        Parameters
        ----------
        reset : boolean, optional
            Determines if the previous fairlead force values will be used for the catenary iteration. The default is False.

        tol : float
            Convergence tolerance for catenary solver measured as absolute error of x and z values in m.
            
        profiles : int
            Values greater than 0 signal for line profile data to be saved (used for plotting, getting distributed tensions, etc).

        Raises
        ------
        LineError
            If the horizontal force at the fairlead (HF) is less than 0

        Returns
        -------
        None.

        '''

        # deal with horizontal tension starting point
        if self.HF < 0:
            raise LineError("Line HF cannot be negative") # this could be a ValueError too...
            
        if reset==True:   # Indicates not to use previous fairlead force values to start catenary 
            self.HF = 0   # iteration with, and insteady use the default values.
        
        
        # ensure line profile information is computed if needed for computing current loads
        if self.sys.currentMod == 1 and profiles == 0:
            profiles = 1

        # get seabed depth and slope under each line end
        depthA, nvecA = self.sys.getDepthFromBathymetry(self.rA[0], self.rA[1])
        depthB, nvecB = self.sys.getDepthFromBathymetry(self.rB[0], self.rB[1])
        
        # deal with height off seabed issues
        if self.rA[2] < -depthA:
            self.rA[2] = -depthA
            self.cb = 0
            #raise LineError("Line {} end A is lower than the seabed.".format(self.number)) <<< temporarily adjust to seabed depth
        elif self.rB[2] < -depthB:
            raise LineError("Line {} end B is lower than the seabed.".format(self.number))
        else:
            self.cb = -depthA - self.rA[2]  # when cb < 0, -cb is defined as height of end A off seabed (in catenary)

        
        # ----- Perform rotation/transformation to 2D plane of catenary -----
        
        dr =  self.rB - self.rA
        
        # if a current force is present, include it in the catenary solution
        if np.sum(np.abs(self.fCurrent)) > 0:
        
            # total line exernal force per unit length vector (weight plus current drag)
            w_vec = self.fCurrent/self.L + np.array([0, 0, -self.type["w"]])
            w = np.linalg.norm(w_vec)
            w_hat = w_vec/w
            
            # get rotation matrix from gravity down to w_vec being down
            if w_hat[0] == 0 and w_hat[1] == 0: 
                if w_hat[2] < 0:
                    R_curr = np.eye(3,3)
                else:
                    R_curr = -np.eye(3,3)
            else:
                R_curr = RotFrm2Vect(w_hat, np.array([0, 0, -1]))  # rotation matrix to make w vertical
        
            # vector from A to B needs to be put into the rotated frame
            dr = np.matmul(R_curr, dr)  
        
        # if no current force, things are simple
        else:
            R_curr = np.eye(3,3)
            w = self.type["w"]
        
        
        # apply a rotation about Z' to align the line profile with the X'-Z' plane
        theta_z = -np.arctan2(dr[1], dr[0])
        R_z = helpers.rotationMatrix(0, 0, theta_z)
        
        # overall rotation matrix (global to catenary plane)
        R = np.matmul(R_z, R_curr)   
        
        # figure out slope in plane (only if contacting the seabed)
        if self.rA[2] <= -depthA or self.rB[2] <= -depthB:
            nvecA_prime = np.matmul(R, nvecA)
        
            dz_dx = -nvecA_prime[0]*(1.0/nvecA_prime[2])  # seabed slope components
            dz_dy = -nvecA_prime[1]*(1.0/nvecA_prime[2])  # seabed slope components
            # we only care about dz_dx since the line is in the X-Z plane in this rotated situation
            alpha = np.degrees(np.arctan(dz_dx))
            cb = self.cb
        else:
            if np.sum(np.abs(self.fCurrent)) > 0 or nvecA[2] < 1: # if there is current or seabed slope
                alpha = 0
                cb = min(0, dr[2]) - 100  # put the seabed out of reach (model limitation)
            else:  # otherwise proceed as usual (this is the normal case)
                alpha = 0
                cb = self.cb
        
        # horizontal and vertical dimensions of line profile (end A to B)
        LH = np.linalg.norm(dr[:2])
        LV = dr[2]
        
        
        # ----- call catenary function or alternative and save results -----
        
        #If EA is found in the line properties we will run the original catenary function 
        if 'EA' in self.type:
            try:
                (fAH, fAV, fBH, fBV, info) = helpers.catenary(LH, LV, self.L, self.EA, w,
                                                      CB=cb, alpha=alpha, HF0=self.HF, VF0=self.VF, 
                                                      Tol=tol, nNodes=self.nNodes, plots=profiles)                                                    
            except CatenaryError as error:
                raise LineError(self.number, error.message)       
        #If EA isnt found then we will use the ten-str relationship defined in the input file 
        else:
            (fAH, fAV, fBH, fBV, info) = helpers.nonlinear(LH, LV, self.L, self.type['Str'], self.type['Ten'],np.linalg.norm(w)) 
    
    
        # save line profile coordinates in global frame (involves inverse rotation)
        if profiles > 0:
            # note: instantiating new arrays rather than writing directly to self.Xs 
            # seems to be necessary to avoid plots auto-updating to the current 
            # profile of the Line object.
            Xs = np.zeros(self.nNodes)
            Ys = np.zeros(self.nNodes)
            Zs = np.zeros(self.nNodes)
            # apply inverse rotation to node positions
            for i in range(0,self.nNodes):
                temp_array = np.array([info['X'][i], 0 ,info['Z'][i]])
                unrot_pos = np.matmul(temp_array, R)
                
                Xs[i] = self.rA[0] + unrot_pos[0]
                Ys[i] = self.rA[1] + unrot_pos[1]
                Zs[i] = self.rA[2] + unrot_pos[2]

            self.Xs = Xs
            self.Ys = Ys
            self.Zs = Zs
            self.Ts = info["Te"]
        
        # save fairlead tension components for use as ICs next iteration
        self.HF = info["HF"]
        self.VF = info["VF"]
        
        # save other important info
        self.LBot = info["LBot"]
        self.z_extreme = self.rA[2] + info["Zextreme"]
        self.info = info
        
        # save forces in global reference frame
        self.fA = np.matmul(np.array([fAH, 0, fAV]), R)
        self.fB = np.matmul(np.array([fBH, 0, fBV]), R)
        self.TA = np.linalg.norm(self.fA) # end tensions
        self.TB = np.linalg.norm(self.fB)
        
        # Compute transverse (out-of-plane) stiffness term
        if fAV > fAH:  # if line is more vertical than horizontal, 
            Kt = 0.5*(fAV-fBV)/LV  # compute Kt based on vertical tension/span
        else:  # otherwise use the classic horizontal approach
            Kt = -fBH/LH
        
        
        # save 3d stiffness matrix in global orientation for both line ends (3 DOF + 3 DOF)
        self.KA  = from2Dto3Drotated(info['stiffnessA'],  Kt, R.T)  # reaction at A due to motion of A
        self.KB  = from2Dto3Drotated(info['stiffnessB'],  Kt, R.T)  # reaction at B due to motion of B
        self.KBA = from2Dto3Drotated(info['stiffnessBA'], Kt, R.T)  # reaction at B due to motion of A
        
        
        # ----- calculate current loads if applicable, for use next time -----
        
        if self.sys.currentMod == 1: 

            U = self.sys.current  # 3D current velocity [m/s]  (could be changed to depth-dependent profile)
            
            fCurrent = np.zeros(3)  # total current force on line in x, y, z [N]        
            
            # Loop through each segment along the line and add up the drag forces.
            # This is in contrast to MoorDyn calculating for nodes.
            for i in range(self.nNodes-1):
                #For each segment find the tangent vector and then calculate the current loading
                dr_seg = np.array([self.Xs[i+1] - self.Xs[i], 
                                   self.Ys[i+1] - self.Ys[i], 
                                   self.Zs[i+1] - self.Zs[i]])  # segment vector
                ds_seg = np.linalg.norm(dr_seg)
                
                if ds_seg > 0:                   # only include if segment length > 0
                    q = dr_seg/ds_seg
                    # transverse and axial current velocity components
                    Uq = np.dot(U, q) * q
                    Up = U - Uq          
                    # transverse and axial drag forces on segment
                    dp = 0.5*self.sys.rho*self.type["Cd"]        *self.type["d_vol"]*ds_seg*np.linalg.norm(Up)*Up
                    dq = 0.5*self.sys.rho*self.type["CdAx"]*np.pi*self.type["d_vol"]*ds_seg*np.linalg.norm(Uq)*Uq
                    # add to total current force on line
                    fCurrent += dp + dq    
            
            self.fCurrent = fCurrent  # save for use next call
        else:
            self.fCurrent = np.zeros(3)  # if no current, ensure this force is zero


        # ----- plot the profile if requested -----
        if profiles > 1:
            import matplotlib.pyplot as plt
            plt.plot(self.info['X'], self.info['Z'])
            plt.show()
    

    def getTension(self, s):
        '''Returns tension at a given point along the line
        
        Parameters
        ----------
        
        s : scalar or array-like
            Value or array of values for the arc length along the line from end A to end B at which
            the information is desired. Positive values are arc length in m, negative values are a
            relative location where 0 is end A, -1 is end B, and -0.5 is the midpoint.
        
        Returns
        -------
        
        tension value(s)
        
        '''
        #if s < 0:
        #    s = -s*self.L            
        #if s > self.L:
        #    raise ValueError('Specified arc length is larger than the line unstretched length.')
        
        Te = np.interp(s, self.info['s'], self.info['Te'])
        
        return Te


    def getPosition(self, s):
        '''Returns position at a given point along the line
        
        Parameters
        ----------
        
        s : scalar or array-like
            Value or array of values for the arc length along the line from end A to end B at which
            the information is desired. Positive values are arc length in m, negative values are a
            relative location where 0 is end A, -1 is end B, and -0.5 is the midpoint.
        
        Returns
        -------
        
        position vector(s)
        
        '''
        
        # >>> should be merged with getLineCoords and getCoordinate functionality <<<
        
        x = np.interp(s, self.info['s'], self.info['X'])
        z = np.interp(s, self.info['s'], self.info['Z'])
        
        
        dr =  self.rB - self.rA                 
        LH = np.hypot(dr[0], dr[1])
        Xs = self.rA[0] + x*dr[0]/LH
        Ys = self.rA[1] + x*dr[1]/LH
        Zs = self.rA[2] + z
        
        return np.vstack([ Xs, Ys, Zs])

    
    def getCost(self):
        '''Fill in a cost dictionary and return the total cost for this Line object.'''
        self.cost = {}  # clear any old cost numbers
        self.cost['material'] = self.type['cost']*self.L0
        total_cost = sum(self.cost.values()) 
        return total_cost
    
    
    def attachLine(self, lineID, endB):
        '''Adds a Line end to the rod

        Parameters
        ----------
        lineID : int
            The identifier ID number of a line
        endB : boolean
            Determines which end of the line is attached to the point

        Returns
        -------
        None.

        '''
        self.attached.append(lineID)  
        self.attachedEndB.append(endB)


    def activateDynamicStiffness(self, display=0):
        '''Switch mooring line model to dynamic line stiffness
        value, including potential unstretched line length
        adjustment. This only works when dynamic line properties
        are used.'''
        
        if self.type['EAd'] > 0:
            # switch to dynamic stiffness value
            EA_old = self.type['EA']
            EA_new = self.type['EAd'] + self.type['EAd_Lm']*np.mean([self.TA, self.TB])  # this implements the sloped Krd = alpha + beta*Lm
            self.EA = np.max([EA_new, EA_old])  # only if the dynamic stiffness is higher than the static stiffness, activate the dynamic stiffness
            
            # adjust line length to maintain current tension (approximate)
            self.L = self.L0 * (1 + self.TB/EA_old)/(1 + self.TB/EA_new)
            
        else:
            if display > 0:
                print(f'Line {self.number} has zero dynamic stiffness coefficient so activateDynamicStiffness does nothing.')
        
    
    def revertToStaticStiffness(self):
        '''Switch mooring line model to dynamic line stiffness
        values, including potential unstretched line length
        adjustment. This only works when dynamic line properties
        are used.'''
        
        # switch to static/default stiffness value
        self.EA = self.type['EA']
        
        # revert to original line length
        self.L = self.L0
    

def from2Dto3Drotated(K2D, Kt, R): 
    '''Initialize a line end's analytic stiffness matrix in the 
    plane of the catenary then rotate the matrix to be about the 
    global frame using [K'] = [R][K][R]^T
    
    Parameters
    ----------
    K2D : 2x2 matrix
        Planar stiffness matrix of line end [N/m]
    Kt : float
        Transverse (out-of-plane) stiffness term [N/m].
    R : 3x3 matrix
        Rotation matrix from global frame to the local
        X-Z plane of the line
        
    Returns
    -------
    3x3 stiffness matrix in global orientation [N/m].
    '''    
    K2 = np.array([[K2D[0,0], 0 , K2D[0,1]],
                   [  0     , Kt,   0     ],
                   [K2D[1,0], 0 , K2D[1,1]]])
    
    return np.matmul(np.matmul(R, K2), R.T)    


def RotFrm2Vect( A, B):
    '''Rodriguez rotation function, which returns the rotation matrix 
    that transforms vector A into Vector B.
    '''
    
    v = np.cross(A,B)
    ssc = np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])
         
    R =  np.eye(3,3) + ssc + np.matmul(ssc,ssc)*(1-np.dot(A,B))/(np.linalg.norm(v)*np.linalg.norm(v))            

    return R


class Point():
    '''A class for any object in the mooring system that can be described by three translational coorindates'''
    
    def __init__(self, mooringSys, num, ptype, r, rod_attachment = '', outs = '', m=0, v=0, fExt=np.zeros(3), DOFs=[0,1,2], d=0, zSpan=[-1,1], CdA=0.0, Ca=0.0):
        '''Initialize Point attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the point object
        num : int
            indentifier number
        ptype : int
            the point type: 0 free to move, 1 fixed, -1 coupled externally
        r : array
            x,y,z coorindate position vector [m].
        m : float, optional
            mass [kg]. The default is 0.
        v : float, optional
            submerged volume [m^3]. The default is 0.
        CdA : float, optional
            Product of drag coefficient and cross sectional area in any direction [m^2]. The default is 0.
        Ca : float, optional
            Added mass coefficient in any direction.
        fExt : array, optional
            applied external force vector in global orientation (not including weight/buoyancy). The default is np.zeros(3).
        DOFs: list
            list of which coordinate directions are DOFs for this point (default 0,1,2=x,y,z). E.g. set [2] for vertical motion only.
        d : float, optional
            diameter [m]. The default is 0.
        zSpan : [float, float], optional
            The lower and upper limits of the Point's volume relative to its coordinate [m]. 
            This only affects the change in buoyancy when crossing the free surface. The 
            default is [-1,1], i.e. a 2-m tall volume.
        
        Returns
        -------
        None.

        '''
    
        self.sys    = mooringSys        # store a reference to the overall mooring system (instance of System class)
    
        self.number = num
        if isinstance(ptype, dict):
            self.type = ptype
            ptype = 0
        else:
            self.type = ptype                # 1: fixed/attached to something, 0 free to move, or -1 coupled externally
        self.r = np.array(r, dtype=np.float_)
        self.entity = {ptype:''}         # dict for entity (e.g. anchor) info
        self.cost = {}                  # empty dictionary to contain cost info
        self.loads = {}                 # empty dictionary to contain load info
        
        self.m  = float(m)
        self.v  = float(v)
        self.CdA= float(CdA)
        self.Ca = float(Ca)
        self.fExt = fExt                # external forces plus weight/buoyancy
        self.fBot = 10.0                # this is a seabed contact force that will be added if a point is specified below the seabed
        self.zSub = 0.0                 # this is the depth that the point is positioned below the seabed (since r[2] will be capped at the depth)
        self.zTol = 2.0                 # depth tolerance to be used when updating the point's position relative to the seabed
        
        self.DOFs = DOFs
        self.nDOF = len(DOFs)
        
        self.d = d                      # the diameter of the point, if applicable. Used for hydrostatics [m]
        
        self.attached     = []         # ID numbers of any Lines attached to the Point
        self.attachedEndB = []         # specifies which end of the line is attached (1: end B, 0: end A)

        self.rod_attachment = rod_attachment
        self.outs = outs
    
        if len(zSpan)==2:
            self.zSpan = np.array(zSpan, dtype=float)
        else:
            raise ValueError("Point zSpan parameter must contain two numbers.")
    
        #print("Created Point "+str(self.number))
    
    
    def attachLine(self, lineID, endB):
        '''Adds a Line end to the Point

        Parameters
        ----------
        lineID : int
            The identifier ID number of a line
        endB : boolean
            Determines which end of the line is attached to the point

        Returns
        -------
        None.

        '''
    
        self.attached.append(lineID)  
        self.attachedEndB.append(endB)
        #print("attached Line "+str(lineID)+" to Point "+str(self.number))
    
    def detachLine(self, lineID, endB):
        '''Detaches a Line end from the Point

        Parameters
        ----------
        lineID : int
            The identifier ID number of a line
        endB : boolean
            Determines which end of the line is to be detached from the point

        Returns
        -------
        None.

        '''
        
        self.attached.pop(self.attached.index(lineID))
        self.attachedEndB.pop(self.attachedEndB.index(endB))
        print("detached Line "+str(lineID)+" from Point "+str(self.number))
    
    
    def setPosition(self, r):
        '''Sets the position of the Point, along with that of any dependent objects.

        Parameters
        ----------
        r : array
            x,y,z coordinate position vector of the point [m]

        Raises
        ------
        ValueError
            If the length of the input r array is not of length 3

        Returns
        -------
        None.

        '''
        
        # update the position of the Point itself
        if len(r) == 3:   # original case, setting all three coordinates as normal, asuming x,y,z
            self.r = np.array(r)
        elif len(r) == self.nDOF:
            self.r[self.DOFs] = r          # this does a mapping based on self.DOFs, to support points with e.g. only a z DOF or only x and z DOFs
        else:
            raise ValueError(f"Point setPosition method requires an argument of size 3 or nDOF, but size {len(r):d} was provided")
        
        # update the point's depth and position based on relation to seabed
        depth, _ = self.sys.getDepthFromBathymetry(self.r[0], self.r[1]) 
        
        self.zSub = np.max([-self.zTol, -self.r[2] - depth])   # depth of submergence in seabed if > -zTol
        self.r = np.array([self.r[0], self.r[1], np.max([self.r[2], -depth])]) # don't let it sink below the seabed
        
        # update the position of any attached Line ends
        for LineID,endB in zip(self.attached,self.attachedEndB):
            self.sys.lineList[LineID-1].setEndPosition(self.r, endB)
            
        if len(self.r) < 3:
            print("Double check how this point's position vector is calculated")
            breakpoint()
            
            
    
    def getForces(self, lines_only=False, seabed=True, xyz=False):
        '''Sums the forces on the Point, including its own plus those of any attached Lines.

        Parameters
        ----------
        lines_only : boolean, optional
            An option for calculating forces from just the mooring lines or not. The default is False.
        seabed : bool, optional
            if False, will not include the effect of the seabed pushing the point up
        xyz : boolean, optional
            if False, returns only forces corresponding to enabled DOFs. If true, returns forces in x,y,z regardless of DOFs. 
            
        Returns
        -------
        f : array
            The force vector applied to the point in its current position [N]

        '''
    
        f = np.zeros(3)         # create empty force vector on the point
        
        if lines_only==False:
            '''
            radius = self.d/2           # can do this, or find the radius using r=(3*self.v/(4*np.pi))**(1/3)
            x = max(0, radius**2 - self.r[2]**2)
            dWP = 2*np.sqrt(x)          # diameter at the waterplane [m]
            AWP = np.pi/4 * dWP**2      # waterplane area [m]
            #v_half = (4/3)*np.pi*(np.sqrt(x)**3) * 0.5  # volume of the half sphere that is cut by the waterplane [m^3]
            #v = abs(-min(0, np.sign(self.r[2]))*self.v - v_half)    # submerged volume of the point [m^3]
            '''
            f[2] += -self.m*self.sys.g  # add weight 

            #f[2] += self.v*self.sys.rho*self.sys.g   # add buoyancy using submerged volume
            
            if self.r[2] + self.zSpan[1] < 0.0:                # add buoyancy if fully submerged
                f[2] +=  self.v*self.sys.rho*self.sys.g
            elif self.r[2] + self.zSpan[0] < 0.0:    # add some buoyancy if part-submerged (linear variation, constant Awp)
                f[2] +=  self.v*self.sys.rho*self.sys.g * (self.r[2] + self.zSpan[0])/(self.zSpan[0]-self.zSpan[1])
            # (no buoyancy force added if it's fully out of the water, which would be very exciting for the Point)
            
            f += np.array(self.fExt) # add external forces
            #f[2] -= self.sys.rho*self.sys.g*AWP*self.r[2]   # hydrostatic heave stiffness
            
            # handle case of Point resting on or below the seabed, to provide a restoring force
            # add smooth transition to fz=0 at seabed (starts at zTol above seabed)
            f[2] += max(self.m - self.v*self.sys.rho, 0)*self.sys.g * (self.zSub + self.zTol)/self.zTol

                
        # add forces from attached lines
        for LineID,endB in zip(self.attached,self.attachedEndB):
            # f += self.sys.lineList[LineID-1].getEndForce(endB)
            if endB:
                f += self.sys.lineList[LineID-1].fB
            else:
                f += self.sys.lineList[LineID-1].fA
        
        if xyz:
            return f
        else:
            return f[self.DOFs]    # return only the force(s) in the enable DOFs
        
    
    
    def getStiffness(self, X = [], tol=0.0001, dx = 0.01):
        '''Gets the stiffness matrix of the point due only to mooring lines with all other objects free to equilibrate.
        NOTE: This method currently isn't set up to worry about nDOF and DOFs settings of the Point. It only works for DOFs=[0,1,2].

        Parameters
        ----------
        X1 : array
            The position vector of the Point at which the stiffness matrix is to be calculated.
        dx : float, optional
            The change in displacement to be used for calculating the change in force. The default is 0.01.

        Returns
        -------
        K : matrix
            The stiffness matrix of the point at the given position X1.

        '''
        
        #print("Getting Point "+str(self.number)+" stiffness matrix...")
        
        if len(X) == 3:
            X1 = np.array(X)
        elif len(X)==0:
            X1 = self.r
        else:
            raise ValueError('Point.getStiffness expects the optional X parameter to be size 3')
        
        # set this Point's type to fixed so mooring system equilibrium response to its displacements can be found
        type0 = self.type                         # store original type to restore later
        self.type = 1                             # set type to 1 (not free) so that it won't be adjusted when finding equilibrium
        
        # if this Point is attached to a Body, set that Body's type to fixed so equilibrium can be found
        for body in self.sys.bodyList:            # search through all the bodies in the mooring system
            if self.number in body.attachedP:     # find the one that this Point is attached to (if at all)
                num = body.number                 # store body number to index later
                Btype0 = body.type                # store original body type to restore later
                body.type = 1                     # set body type to 1 (not free) so that it won't be adjusted when finding equilibrium 
        
        # ensure this Point is positioned at the desired linearization point
        self.setPosition(X1)                      # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)       # find equilibrium of mooring system given this Point in current position
        f = self.getForces(lines_only=True)       # get the net 6DOF forces/moments from any attached lines 

        # Build a stiffness matrix by perturbing each DOF in turn
        K = np.zeros([3,3])
        
        for i in range(len(K)):
            X2 = X1 + np.insert(np.zeros(2),i,dx) # calculate perturbed Point position by adding dx to DOF in question            
            self.setPosition(X2)                  # perturb this Point's position
            self.sys.solveEquilibrium3(tol=tol)   # find equilibrium of mooring system given this Point's new position
            f_2 =self.getForces(lines_only=True)  # get the net 3DOF forces/moments from any attached lines 

            K[:,i] = -(f_2-f)/dx                  # get stiffness in this DOF via finite difference and add to matrix column
            
        # ----------------- restore the system back to previous positions ------------------
        self.setPosition(X1)                      # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)       # find equilibrium of mooring system given this Point in current position
        self.type = type0                         # restore the Point's type to its original value
        for body in self.sys.bodyList:
            if self.number in body.attachedP:
                num = body.number
                self.sys.bodyList[num-1].type = Btype0    # restore the type of the Body that the Point is attached to back to original value

        
        return K
    
    
    
    def getStiffnessA(self, lines_only=False, xyz=False):
        '''Gets analytical stiffness matrix of Point due only to mooring lines with other objects fixed.

        Returns
        -------
        K : matrix
            3x3 analytic stiffness matrix.

        '''
        
        #print("Getting Point "+str(self.number)+" analytic stiffness matrix...")
        
        K = np.zeros([3,3])         # create an empty 3x3 stiffness matrix
        
        # append the stiffness matrix of each line attached to the point
        for lineID,endB in zip(self.attached,self.attachedEndB):
            line = self.sys.lineList[lineID-1]
            #KA, KB = line.getStiffnessMatrix()
            
            if endB == 1:                  # assuming convention of end A is attached to the point, so if not,
                #KA, KB = KB, KA            # swap matrices of ends A and B                                
                K += line.KB
            else:
                K += line.KA 
            
        # NOTE: can rotate the line's stiffness matrix in either Line.getStiffnessMatrix() or here in Point.getStiffnessA()
        
        # add seabed or hydrostatic terms if needed
        if lines_only==False:
        
            # if partially submerged, apply a hydrostatic stiffness based on buoyancy
            if self.r[2] + self.zSpan[1] > 0.0 and self.r[2] + self.zSpan[0] < 0.0: 
                K[2,2] += self.sys.rho*self.sys.g * self.v/(self.zSpan[1]-self.zSpan[0])  # assumes volume is distributed evenly across zSpan
            
            # if less than zTol above the seabed (could even be below the seabed), apply a stiffness (should bring wet weight to zero at seabed)
            if self.r[2] < self.zTol - self.sys.depth:
                K[2,2] += max(self.m - self.v*self.sys.rho, 0)*self.sys.g / self.zTol
                
            # if on seabed, apply a large stiffness to help out system equilibrium solve (if it's transitioning off, keep it a small step to start with)    
            if self.r[2] == -self.sys.depth:
                K[2,2] += 1.0e12
        if sum(np.isnan(K).ravel()) > 0: breakpoint()
        if xyz:                     # if asked to output all DOFs, do it
            return K
        else:                       # otherwise only return rows/columns of active DOFs
            return K[:,self.DOFs][self.DOFs,:]
        
    
    def getCost(self):
        '''Fill in and returns a cost dictionary for this Point object.
        So far it only applies for if the point is an anchor.'''
        
        from moorpy.MoorProps import getAnchorCost
        
        self.cost = {'material':0}  # clear any old cost numbers and start with 0
        
        # figure out if it should be an anchor if it isn't already defined
        if self.entity['type'] == '':
            depth, _ = self.sys.getDepthFromBathymetry(self.r[0], self.r[1]) 
            if self.r[3] == depth and self.type==1:  # if it's fixed on the seabed
                self.entity['type'] = 'anchor'       # assume it's an anchor
                if self.FA[2] == 0:
                    self.entity['anchor_type'] = 'drag-embedment'
                else:
                    self.entity['anchor_type'] = 'suction'
        
        # calculate costs if it's an anchor (using simple model)
        if self.entity['type'] == 'anchor':
            self.cost['material'] = getAnchorCost(self.loads['fx_max'], 
                                                  self.loads['fz_max'],
                                             type=self.entity['anchor_type'])
        
        return self.cost    

class Body():
    '''A class for any object in the mooring system that will have its own reference frame'''
    
    def __init__(self, mooringSys, num, type, r6, m=0, v=0, rCG=np.zeros(3),
                 AWP=0, rM=np.zeros(3), f6Ext=np.zeros(6), I=np.zeros(3),
                 CdA=np.zeros(3), Ca=np.zeros(3), DOFs=[0,1,2,3,4,5]):
        '''Initialize Body attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the body object
        num : int
            indentifier number
        type : int
            the body type: 0 free to move, 1 fixed, -1 coupled externally
        r6 : array
            6DOF position and orientation vector [m, rad]
        m : float, optional
            mass, centered at CG [kg]. The default is 0.
        v : float, optional
            volume, centered at reference point [m^3]. The default is 0.
        rCG : array, optional
            center of gravity position in body reference frame [m]. The default is np.zeros(3).
        AWP : float, optional
            waterplane area - used for hydrostatic heave stiffness if nonzero [m^2]. The default is 0.
        rM : float or array, optional
            coorindates or height of metacenter relative to body reference frame [m]. The default is np.zeros(3).
        f6Ext : array, optional
            applied external forces and moments vector in global orientation (not including weight/buoyancy) [N]. The default is np.zeros(6).
        I : array, optional
            Mass moment of inertia about 3 axes.
        CdA : array, optional
            Product of drag coefficient and frontal area in three directions [m^2].
        Ca : array, optional
            Added mass coefficient in three directions.
        DOFs: list, optional
            list of the DOFs for this body (0=surge,1=sway,...5=yaw). Any not 
            listed will be held fixed. E.g. [0,1,5] for 2D horizontal motion.
        
        Returns
        -------
        None.

        '''
    
        self.sys    = mooringSys       # store a reference to the overall mooring system (instance of System class)
        
        self.number = num
        self.type   = type                          # 0 free to move, or -1 coupled externally
        self.r6     = np.array(r6, dtype=np.float_)     # 6DOF position and orientation vector [m, rad]
        self.m      = m                             # mass, centered at CG [kg]
        self.v      = v                             # volume, assumed centered at reference point [m^3]
        self.rCG    = np.array(rCG, dtype=np.float_)    # center of gravity position in body reference frame [m]
        self.AWP    = AWP                           # waterplane area - used for hydrostatic heave stiffness if nonzero [m^2]
        if np.isscalar(rM):
            self.rM = np.array([0,0,rM], dtype=np.float_) # coordinates of body metacenter relative to body reference frame [m]
        else:
            self.rM = np.array(rM, dtype=np.float_)       

        # >>> should streamline the below <<<
        if np.isscalar(I):
            self.I = np.array([I,I,I], dtype=float)
        else:
            self.I = np.array(I, dtype=float)    

        if np.isscalar(CdA):
            self.CdA = np.array([CdA,CdA,CdA], dtype=float)
        else:
            self.CdA = np.array(CdA, dtype=float)    
            
        if np.isscalar(Ca):
            self.Ca = np.array([Ca,Ca,Ca], dtype=float) 
        else:
            self.Ca = np.array(Ca, dtype=float)                
                
        self.f6Ext  = np.array(f6Ext, dtype=float)    # for adding external forces and moments in global orientation (not including weight/buoyancy)
        
        self.DOFs = DOFs
        self.nDOF = len(DOFs)
        
        self.attachedP   = []          # ID numbers of any Points attached to the Body
        self.rPointRel   = []          # coordinates of each attached Point relative to the Body reference frame
        
        self.attachedR   = []          # ID numbers of any Rods attached to the Body (not yet implemented)
        self.r6RodRel   = []           # coordinates and unit vector of each attached Rod relative to the Body reference frame
        
        self.R = np.eye(3)             # body orientation rotation matrix
        #print("Created Body "+str(self.number))
        
    
    def attachPoint(self, pointID, rAttach):
        '''Adds a Point to the Body, at the specified relative position on the body.
        
        Parameters
        ----------
        pointID : int
            The identifier ID number of a point
        rAttach : array
            The position of the point relative to the body's frame [m]

        Returns
        -------
        None.

        '''
    
        self.attachedP.append(pointID)
        self.rPointRel.append(np.array(rAttach))
        
        if self.sys.display > 1:
            print("attached Point "+str(pointID)+" to Body "+str(self.number))
    
    
    def attachRod(self, rodID, endCoords):
        '''Adds a Point to the Body, at the specified relative position on the body.
        
        Parameters
        ----------
        rodID : int
            The identifier ID number of a point
        endCoords : array
            The position of the Rods two ends relative to the body reference frame [m]

        Returns
        -------
        None.

        '''
    
        k = (endCoords[3:]-endCoords[:3])/np.linalg.norm(endCoords[3:]-endCoords[:3])
    
        self.attachedR.append(rodID)
        self.r6RodRel.append(np.hstack([ endCoords[:3], k]))
        
        print("attached Rod "+str(rodID)+" to Body "+str(self.number))
        
    
    def setPosition(self, r6):
        '''Sets the position of the Body, along with that of any dependent objects.

        Parameters
        ----------
        r6 : array
            6DOF position and orientation vector of the body [m, rad]

        Raises
        ------
        ValueError
            If the length of the input r6 array is not of length 6

        Returns
        -------
        None.

        '''
        
        # update the position/orientation of the body
        if len(r6)==6:
            self.r6 = np.array(r6, dtype=np.float_)  # update the position of the Body itself
        elif len(r6) == self.nDOF:
            self.r6[self.DOFs] = r6  # mapping to use only some DOFs
        else:
            raise ValueError(f"Body setPosition method requires an argument of size 6 or nDOF, but size {len(r6):d} was provided")
        
        self.R = helpers.rotationMatrix(self.r6[3], self.r6[4], self.r6[5])   # update body rotation matrix
        
        # update the position of any attached Points
        for PointID,rPointRel in zip(self.attachedP,self.rPointRel):
            rPoint = np.matmul(self.R, rPointRel) + self.r6[:3]  # rPoint = transformPosition(rPointRel, r6)            
            self.sys.pointList[PointID-1].setPosition(rPoint)
            
        # update the position of any attached Rods        
        for rodID,r6Rel in zip(self.attachedR,self.r6RodRel):        
            rA = np.matmul(self.R, r6Rel[:3]) + self.r6[:3] 
            k = np.matmul(self.R, r6Rel[3:])
            self.sys.rodList[rodID-1].rA = rA
            self.sys.rodList[rodID-1].rB = rA + k*self.sys.rodList[rodID-1].L
   
        if self.sys.display > 3:     
            helpers.printVec(rPoint)
            breakpoint()
            
        
   
    def getForces(self, lines_only=False, all_DOFs=False):
        '''Sums the forces and moments on the Body, including its own plus those from any attached objects.
        Forces and moments are aligned with global x/y/z directions but are relative 
        to the body's local reference point.

        Parameters
        ----------
        lines_only : boolean, optional
            If true, the Body's contribution to the forces is ignored.
        all_DOFs : boolean, optional
            True: return all forces/moments; False: only those in DOFs list.

        Returns
        -------
        f6 : array
            The 6DOF forces and moments applied to the body in its current position [N, Nm]
        '''
    
        f6 = np.zeros(6)
    
        # TODO: could save time in below by storing the body's rotation matrix when it's position is set rather than 
        #       recalculating it in each of the following function calls.
                
        if lines_only==False:
        
            # add weight, which may result in moments as well as a force
            rCG_rotated = helpers.rotatePosition(self.rCG, self.r6[3:])                     # relative position of CG about body ref point in unrotated reference frame  
            f6 += helpers.translateForce3to6DOF(rCG_rotated, np.array([0,0, -self.m*self.sys.g]))    # add to net forces/moments
        
            # add buoyancy force and moments if applicable (this can include hydrostatic restoring moments 
            # if rM is considered the metacenter location rather than the center of buoyancy)
            rM_rotated = helpers.rotatePosition(self.rM, self.r6[3:])                       # relative position of metacenter about body ref point in unrotated reference frame  
            f6 += helpers.translateForce3to6DOF(rM_rotated, np.array([0,0, self.sys.rho*self.sys.g*self.v]))  # add to net forces/moments
        
            # add hydrostatic heave stiffness (if AWP is nonzero)
            f6[2] -= self.sys.rho*self.sys.g*self.AWP*self.r6[2]
            
            # add any externally applied forces/moments (in global orientation)
            f6 += self.f6Ext
	
        # add forces from any attached Points (and their attached lines)
        for PointID,rPointRel in zip(self.attachedP,self.rPointRel):
        
            fPoint = self.sys.pointList[PointID-1].getForces(lines_only=lines_only) # get net force on attached Point
            rPoint_rotated = helpers.rotatePosition(rPointRel, self.r6[3:])                 # relative position of Point about body ref point in unrotated reference frame  
            f6 += helpers.translateForce3to6DOF(rPoint_rotated, fPoint)                     # add net force and moment resulting from its position to the Body
            
            
        # All forces and moments on the body should now be summed, and are in global/unrotated orientations.
        '''
        # For application to the body DOFs, convert the moments to be about the body's local/rotated x/y/z axes <<< do we want this in all cases? 
        rotMat = rotationMatrix(*self.r6[3:])                                       # get rotation matrix for body
        moment_about_body_ref = np.matmul(rotMat.T, f6[3:])                         # transform moments so that they are about the body's local/rotated axes
        f6[3:] = moment_about_body_ref                                              # use these moments
        '''
        
        if all_DOFs:
            return f6
        else:
            return f6[self.DOFs]

    
    def getStiffness(self, X = [], tol=0.0001, dx = 0.1, all_DOFs=False):
        '''Gets the stiffness matrix of a Body due only to mooring lines with all other objects free to equilibriate.
        The rotational indices of the stiffness matrix correspond to the global x/y/z directions.
        
        Parameters
        ----------
        X1 : array
            The position vector (6DOF) of the main axes of the Body at which the stiffness matrix is to be calculated.
        dx : float, optional
            The change in displacement to be used for calculating the change in force. The default is 0.01.
        all_DOFs : boolean, optional
            True: return all forces/moments; False: only those in DOFs list.

        Returns
        -------
        K : matrix
            The stiffness matrix of the body at the given position X1.
            
        '''
        
        #print("Getting Body "+str(self.number)+" stiffness matrix...")
        
        if len(X) == 6:
            X1 = np.array(X)
        elif len(X)==0:
            X1 = self.r6
        else:
            raise ValueError('Body.getStiffness expects the optional X parameter to be size 6')
        
        # set this Body's type to fixed so mooring system equilibrium response to its displacements can be found
        type0 = self.type                         # store original type to restore later
        self.type = 1                             # set type to 1 (not free) so that it won't be adjusted when finding equilibrium
        
        # ensure this Body is positioned at the desired linearization point
        self.setPosition(X1)                      # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)       # find equilibrium of mooring system given this Body in current position
        f6 = self.getForces(lines_only=True)      # get the net 6DOF forces/moments from any attached lines 
        
        # Build a stiffness matrix by perturbing each DOF in turn
        K = np.zeros([6,6])
        
        for i in range(len(K)):
            X2 = X1 + np.insert(np.zeros(5),i,dx) # calculate perturbed Body position by adding dx to DOF in question            
            self.setPosition(X2)                  # perturb this Body's position
            self.sys.solveEquilibrium3(tol=tol)   # find equilibrium of mooring system given this Body's new position
            f6_2 =self.getForces(lines_only=True) # get the net 6DOF forces/moments from any attached lines 
            
            K[:,i] = -(f6_2-f6)/dx                # get stiffness in this DOF via finite difference and add to matrix column
            
        # ----------------- restore the system back to previous positions ------------------
        self.setPosition(X1)                      # set position to linearization point
        self.sys.solveEquilibrium3(tol=tol)       # find equilibrium of mooring system given this Body in current position
        self.type = type0                         # restore the Body's type to its original value
        
        # Return stiffness matrix
        if all_DOFs:
            return K
        else:  # only return rows/columns of active DOFs
            return K[:,self.DOFs][self.DOFs,:]
        
    
    def getStiffnessA(self, lines_only=False, all_DOFs=False):
        '''Gets the analytical stiffness matrix of the Body with other objects fixed.

        Parameters
        ----------
        lines_only : boolean, optional
            If true, the Body's contribution to its stiffness is ignored.
        all_DOFs : boolean, optional
            True: return all forces/moments; False: only those in DOFs list.

        Returns
        -------
        K : matrix
            6x6 analytic stiffness matrix.
        '''
                
        K = np.zeros([6,6])
        
        
        # stiffness contributions from attached points (and any of their attached lines)
        for PointID,rPointRel in zip(self.attachedP,self.rPointRel):
            
            r = helpers.rotatePosition(rPointRel, self.r6[3:])          # relative position of Point about body ref point in unrotated reference frame  
            f3 = self.sys.pointList[PointID-1].getForces()      # total force on point (for additional rotational stiffness term due to change in moment arm)
            K3 = self.sys.pointList[PointID-1].getStiffnessA()  # local 3D stiffness matrix of the point
            
            # following are from functions translateMatrix3to6
            H = helpers.getH(r)
            K[:3,:3] += K3
            K[:3,3:] += np.matmul(K3, H)                        # only add up one off-diagonal sub-matrix for now, then we'll mirror at the end
            K[3:,3:] += -np.matmul(helpers.getH(f3), H) - np.matmul(H, np.matmul(K3,H))   # updated 2023-05-02
   
        K[3:,:3] = K[:3,3:].T                                   # copy over other off-diagonal sub-matrix
        
        
        # body's own stiffness components
        if lines_only == False:
        
            # rotational stiffness effect of weight
            rCG_rotated = helpers.rotatePosition(self.rCG, self.r6[3:]) # relative position of CG about body ref point in unrotated reference frame  
            Kw = -np.matmul( helpers.getH([0,0, -self.m*self.sys.g]) , helpers.getH(rCG_rotated) )
            
            # rotational stiffness effect of buoyancy at metacenter
            rM_rotated = helpers.rotatePosition(self.rM, self.r6[3:])   # relative position of metacenter about body ref point in unrotated reference frame  
            Kb = -np.matmul( helpers.getH([0,0, self.sys.rho*self.sys.g*self.v]) , helpers.getH(rM_rotated) )
           
            # hydrostatic heave stiffness (if AWP is nonzero)
            Kwp = self.sys.rho*self.sys.g*self.AWP
            
            K[3:,3:] += Kw + Kb
            K[2 ,2 ] += Kwp
            
        # Return stiffness matrix
        if all_DOFs:
            return K
        else:  # only return rows/columns of active DOFs
            return K[:,self.DOFs][self.DOFs,:]
    
        
    
    def draw(self, ax):
        '''Draws the reference axis of the body

        Parameters
        ----------
        ax : axes
            matplotlib.pyplot axes to be used for drawing and plotting.

        Returns
        -------
        linebit : list
            a list to hold plotted lines of the body's frame axes.

        '''
        
        linebit = []  # make empty list to hold plotted lines, however many there are
    
        rx = helpers.transformPosition(np.array([5,0,0]),self.r6)
        ry = helpers.transformPosition(np.array([0,5,0]),self.r6)
        rz = helpers.transformPosition(np.array([0,0,5]),self.r6)
        
        linebit.append(ax.plot([self.r6[0], rx[0]], [self.r6[1], rx[1]], [self.r6[2], rx[2]], color='r'))
        linebit.append(ax.plot([self.r6[0], ry[0]], [self.r6[1], ry[1]], [self.r6[2], ry[2]], color='g'))
        linebit.append(ax.plot([self.r6[0], rz[0]], [self.r6[1], rz[1]], [self.r6[2], rz[2]], color='b'))
        
        self.linebit = linebit
            
        return linebit
    
    
    def redraw(self):
        '''Redraws the reference axis of the body

        Returns
        -------
        linebit : list
            a list to hold redrawn lines of the body's frame axes.

        '''
    
        linebit = self.linebit
    
        rx = helpers.transformPosition(np.array([5,0,0]),self.r6)
        ry = helpers.transformPosition(np.array([0,5,0]),self.r6)
        rz = helpers.transformPosition(np.array([0,0,5]),self.r6)
        
        linebit[0][0].set_data_3d([self.r6[0], rx[0]], [self.r6[1], rx[1]], [self.r6[2], rx[2]])
        linebit[1][0].set_data_3d([self.r6[0], ry[0]], [self.r6[1], ry[1]], [self.r6[2], ry[2]])
        linebit[2][0].set_data_3d([self.r6[0], rz[0]], [self.r6[1], rz[1]], [self.r6[2], rz[2]])
        '''
        linebit[0][0].set_data([self.r6[0], rx[0]], [self.r6[1], rx[1]])
        linebit[0][0].set_3d_properties([self.r6[2], rx[2]])
        linebit[1][0].set_data([self.r6[0], ry[0]], [self.r6[1], ry[1]]) 
        linebit[1][0].set_3d_properties([self.r6[2], ry[2]])
        linebit[2][0].set_data([self.r6[0], rz[0]], [self.r6[1], rz[1]]) 
        linebit[2][0].set_3d_properties([self.r6[2], rz[2]])
        '''
        return linebit
    



class System():
    '''A class for the whole mooring system'''
    
    # >>> note: system module will need to import Line, Point, Body for its add/creation routines 
    #     (but line/point/body modules shouldn't import system) <<<
    
    def __init__(self, file="", dirname="", rootname="", depth=0, rho=1025, g=9.81, qs=1, Fortran=True, lineProps=None, **kwargs):
        '''Creates an empty MoorPy mooring system data structure and will read an input file if provided.

        Parameters
        ----------
        file : string, optional
            An input file, usually a MoorDyn input file, that can be read into a MoorPy system. The default is "".
        depth : float, optional
            Water depth of the system. The default is 0.
        rho : float, optional
            Water density of the system. The default is 1025.
        g : float, optional
            Gravity of the system. The default is 9.81.
        bathymetry : filename, optional
            Filename for MoorDyn-style bathymetry input file.

        Returns
        -------
        None.

        '''
        
        # lists to hold mooring system objects
        self.bodyList = []
        self.rodList = []  # note: Rods are currently only fully supported when plotting MoorDyn output, not in MoorPy modeling
        self.pointList = []
        self.lineList = []
        self.lineTypes = {}
        self.rodTypes = {}
        
        # load mooring line property scaling coefficients for easy use when creating line types
        self.lineProps = helpers.loadLineProps(lineProps)
        
        # the ground body (number 0, type 1[fixed]) never moves but is the parent of all anchored things
        self.groundBody = Body(self, 0, 1, np.zeros(6))   # <<< implementation not complete <<<< be careful here if/when MoorPy is split up
        
        # constants used in the analysis
        self.depth = depth  # water depth [m]
        self.rho   = rho    # water density [kg/m^3]
        self.g     = g      # gravitational acceleration [m/s^2]
        
        # water current - currentMod 0 = no current; 1 = steady uniform current
        self.currentMod = 0         # flag for current model to use
        self.current = np.zeros(3)  # current velocity vector [m/s]
        if 'current' in kwargs:
            self.currentMod = 1
            self.current = helpers.getFromDict(kwargs, 'current', shape=3)
            
        # seabed bathymetry - seabedMod 0 = flat; 1 = uniform slope, 2 = grid
        self.seabedMod = 0
        
        if 'xSlope' in kwargs or 'ySlope' in kwargs:
            self.seabedMod = 1
            self.xSlope = helpers.getFromDict(kwargs, 'xSlope', default=0)
            self.ySlope = helpers.getFromDict(kwargs, 'ySlope', default=0)
        
        if 'bathymetry' in kwargs:
            self.seabedMod = 2
            self.bathGrid_Xs, self.bathGrid_Ys, self.bathGrid = helpers.readBathymetryFile(kwargs['bathymetry'])
        
        
        # initializing variables and lists        
        self.nDOF = 0       # number of (free) degrees of freedom of the mooring system (needs to be set elsewhere)        
        self.freeDOFs = []  # array of the values of the free DOFs of the system at different instants (2D list)
        
        self.nCpldDOF = 0   # number of (coupled) degrees of freedom of the mooring system (needs to be set elsewhere)        
        self.cpldDOFs = []  # array of the values of the coupled DOFs of the system at different instants (2D list)
        
        self.display = 0    # a flag that controls how much printing occurs in methods within the System (Set manually. Values > 0 cause increasing output.)
        
        self.MDoptions = {} # dictionary that can hold any MoorDyn options read in from an input file, so they can be saved in a new MD file if need be
        self.MDoutputs = [] # list of moordyn outputs

        
        # read in data from an input file if a filename was provided
        if len(file) > 0:
            self.load(file)
        
        # set the quasi-static/dynamic toggle for the entire mooring system
        self.qs = qs

    def load(self, filename, clear=True):
            '''Loads a MoorPy System from a MoorDyn-style input file

            Parameters
            ----------
            filename : string
                the file name of a MoorDyn-style input file.
            clear : boolean
                Starts from a clean slate when true. When false, will build on existing mooring system objects.

            Raises
            ------
            ValueError
                DESCRIPTION.

            Returns
            -------
            None.

            '''
            
            # create/empty the lists to start with
            if clear:
                RodDict   = {}  # create empty dictionary for rod types
                self.lineTypes = {}  # create empty dictionary for line types
                self.rodTypes = {}  # create empty dictionary for line types

                self.bodyList = []
                self.rodList  = []
                self.pointList= []
                self.lineList = []

            
            # figure out if it's a YAML file or MoorDyn-style file based on the extension, then open and process
            print('attempting to read '+filename)
            
            # assuming YAML format
            if ".yaml" in filename.lower() or ".yml" in filename.lower():
            
                with open(filename) as file:
                    mooring = yaml.load(file, Loader=yaml.FullLoader)    # get dict from YAML file
            
                self.parseYAML(mooring)

            # assuming normal form
            else: 
                f = open(filename, 'r')

                # read in the data
                
                for line in f:          # loop through each line in the file

                    # get line type property sets
                    if line.count('---') > 0 and (line.upper().count('LINE DICTIONARY') > 0 or line.upper().count('LINE TYPES') > 0):
                        line = next(f) # skip this header line, plus channel names and units lines
                        line = next(f)
                        line = next(f)
                        while line.count('---') == 0:
                            entries = line.split()  # entries: TypeName   Diam    Mass/m     EA     BA/-zeta    EI         Cd     Ca     CdAx    CaAx
                            #self.addLineType(entries[0], float(entries[1]), float(entries[2]), float(entries[3])) 
                            
                            type_string = entries[0]
                            d    = float(entries[1])
                            mass = float(entries[2])
                            w = (mass - np.pi/4*d**2 *self.rho)*self.g                        
                            lineType = dict(name=type_string, d_vol=d, w=w, m=mass)  # make dictionary for this rod type
                            
                            # support linear (EA) or nonlinear (filename string) option for elasticity
                            #if there is a text file in the EA input 
                            if entries[3].find(".txt") != -1:
                                #Then we read in ten-strain file
                                ten_str_fname = entries[3]
                                ten_str = open(ten_str_fname[1:-1], 'r') 
                                
                                #Read line in ten-strain file until we hit '---' signifying the end of the file
                                for line in ten_str:
                                        #skip first 3 lines (Header for input file)
                                        line = next(ten_str)
                                        line = next(ten_str)
                                        line = next(ten_str)
                                        #Preallocate Arrays
                                        str_array = []
                                        ten_array = []
                                        #Loop through lines until you hit '---' signifying the end of the file 
                                        while line.count('---') == 0:
                                            ten_str_entries = line.split() #split entries ten_str_entries: strain tension
                                            str_array.append(ten_str_entries[0]) #First one is strain
                                            ten_array.append(ten_str_entries[1]) #Second one is tension
                                            line = next(ten_str) #go to next line
                                lineType['Str'] = str_array #make new entry in the dictionary to carry tension and strain arrays
                                lineType['Ten'] = ten_array

                            else:

                                try:
                                    lineType['EA'] = float(entries[3].split('|')[0])         # get EA, and only take first value if multiples are given
                                except:
                                    lineType['EA'] = 1e9
                                    print('EA entry not recognized - using placeholder value of 1000 MN')
                            if len(entries) >= 10: # read in other elasticity and hydro coefficients as well if enough columns are provided
                                lineType['BA'  ] = float(entries[4].split('|')[0])
                                lineType['EI'  ] = float(entries[5])
                                lineType['Cd'  ] = float(entries[6])
                                lineType['Ca'  ] = float(entries[7])
                                lineType['CdAx'] = float(entries[8])
                                lineType['CaAx'] = float(entries[9])
                                lineType['material'] = type_string
                            
                            if type_string in self.lineTypes:                         # if there is already a line type with this name
                                self.lineTypes[type_string].update(lineType)          # update the existing dictionary values rather than overwriting with a new dictionary
                            else:
                                self.lineTypes[type_string] = lineType
                            
                            line = next(f)
                            
                            
                    # get rod type property sets
                    if line.count('---') > 0 and (line.upper().count('ROD DICTIONARY') > 0 or line.upper().count('ROD TYPES') > 0):
                        line = next(f) # skip this header line, plus channel names and units lines
                        line = next(f)
                        line = next(f)
                        while line.count('---') == 0:
                            entries = line.split()  # entries: TypeName      Diam     Mass/m    Cd     Ca      CdEnd    CaEnd
                            #RodTypesName.append(entries[0]) # name string
                            #RodTypesD.append(   entries[1]) # diameter
                            #RodDict[entries[0]] = entries[1] # add dictionary entry with name and diameter
                            
                            type_string = entries[0]
                            d    = float(entries[1])
                            mass = float(entries[2])
                            w = (mass - np.pi/4*d**2 *self.rho)*self.g
                            
                            rodType = dict(name=type_string, d_vol=d, w=w, m=mass)  # make dictionary for this rod type
                            
                            if len(entries) >= 7: # read in hydro coefficients as well if enough columns are provided
                                rodType['Cd'   ] = float(entries[3])
                                rodType['Ca'   ] = float(entries[4])
                                rodType['CdEnd'] = float(entries[5])
                                rodType['CaEnd'] = float(entries[6])
                            
                            if type_string in self.rodTypes:                        # if there is already a rod type with this name
                                self.rodTypes[type_string].update(rodType)          # update the existing dictionary values rather than overwriting with a new dictionary
                            else:
                                self.rodTypes[type_string] = rodType
                            
                            line = next(f)
                            
                            
                    # get properties of each Body
                    if line.count('---') > 0 and (line.upper().count('BODIES') > 0 or line.upper().count('BODY LIST') > 0 or line.upper().count('BODY PROPERTIES') > 0):
                        line = next(f) # skip this header line, plus channel names and units lines
                        line = next(f)
                        line = next(f)
                        while line.count('---') == 0:
                            entries = line.split()  # entries: ID   Attachment  X0  Y0  Z0  r0  p0  y0    M  CG*  I*    V  CdA*  Ca*            
                            num = int(entries[0])
                            entry0 = entries[1].lower()                         
                            #num = np.int_("".join(c for c in entry0 if not c.isalpha()))  # remove alpha characters to identify Body #
                            
                            if ("fair" in entry0) or ("coupled" in entry0) or ("ves" in entry0):       # coupled case
                                bodyType = -1                        
                            elif ("fix" in entry0) or ("anchor" in entry0):                            # fixed case
                                bodyType = 1
                            elif ("con" in entry0) or ("free" in entry0):                              # free case
                                bodyType = 0
                            else:                                                                      # for now assuming unlabeled free case
                                bodyType = 0
                                # if we detected there were unrecognized chars here, could: raise ValueError(f"Body type not recognized for Body {num}")
                            #bodyType = -1   # manually setting the body type as -1 for FAST.Farm SM investigation
                            
                            r6  = np.array(entries[2:8], dtype=float)   # initial position and orientation [m, rad]
                            r6[3:] = r6[3:]*np.pi/180.0                 # convert from deg to rad
                            #rCG = np.array(entries[7:10], dtype=float)  # location of body CG in body reference frame [m]
                            m = np.float_(entries[8])                   # mass, centered at CG [kg]
                            v = np.float_(entries[11])                   # volume, assumed centered at reference point [m^3]
                            
                            # process CG
                            strings_rCG = entries[ 9].split("|")                   # split by braces, if any
                            if len(strings_rCG) == 1:                              # if only one entry, it is the z coordinate
                                rCG = np.array([0.0, 0.0, float(strings_rCG[0])])
                            elif len(strings_rCG) == 3:                            # all three coordinates provided
                                rCG = np.array(strings_rCG, dtype=float)
                            else:
                                raise Exception(f"Body {num} CG entry (col 10) must have 1 or 3 numbers.")
                                
                            # process mements of inertia
                            strings_I = entries[10].split("|")                     # split by braces, if any
                            if len(strings_I) == 1:                                # if only one entry, use it for all directions
                                Inert = np.array(3*strings_I, dtype=float)
                            elif len(strings_I) == 3:                              # all three coordinates provided
                                Inert = np.array(strings_I, dtype=float)
                            else:
                                raise Exception(f"Body {num} inertia entry (col 11) must have 1 or 3 numbers.")
                            
                            # process drag ceofficient by area product
                            strings_CdA = entries[12].split("|")                   # split by braces, if any
                            if len(strings_CdA) == 1:                              # if only one entry, use it for all directions
                                CdA = np.array(3*strings_CdA, dtype=float)
                            elif len(strings_CdA) == 3:                            # all three coordinates provided
                                CdA = np.array(strings_CdA, dtype=float)
                            else:
                                raise Exception(f"Body {num} CdA entry (col 13) must have 1 or 3 numbers.")
                            
                            # process added mass coefficient
                            strings_Ca = entries[13].split("|")                    # split by braces, if any				
                            if len(strings_Ca) == 1:                               # if only one entry, use it for all directions
                                Ca = np.array(strings_Ca, dtype=float)
                            elif len(strings_Ca) == 3:                             #all three coordinates provided
                                Ca = np.array(strings_Ca, dtype=float)
                            else:
                                raise Exception(f"Body {num} Ca entry (col 14) must have 1 or 3 numbers.")
                            
                            # add the body
                            self.bodyList.append( Body(self, num, bodyType, r6, m=m, v=v, rCG=rCG, I=Inert, CdA=CdA, Ca=Ca) )
                                        
                            line = next(f)
                            
                            
                    # get properties of each rod
                    if line.count('---') > 0 and (line.upper().count('RODS') > 0 or line.upper().count('ROD LIST') > 0 or line.upper().count('ROD PROPERTIES') > 0):
                        line = next(f) # skip this header line, plus channel names and units lines
                        line = next(f)
                        line = next(f)
                        while line.count('---') == 0:
                            entries = line.split()  # entries: RodID  RodType  Attachment  Xa   Ya   Za   Xb   Yb   Zb  NumSegs  Flags/Outputs
                            num = int(entries[0])
                            rodType = self.rodTypes[entries[1]]
                            attachment = entries[2].lower()
                            dia = rodType['d_vol']  # find diameter based on specified rod type string
                            rA = np.array(entries[3:6], dtype=float)
                            rB = np.array(entries[6:9], dtype=float)
                            nSegs = int(entries[9])
                            outs = entries[10]
                            # >>> note: this is currently only set up for use with MoorDyn output data <<<
                            
                            if nSegs==0:       # this is the zero-length special case
                                lUnstr = 0
                                self.rodList.append( Point(self, num, rodType, rA, outs = outs, rod_attachment=attachment) )
                            else:
                                lUnstr = np.linalg.norm(rB-rA)
                                self.rodList.append( Line(self, num, lUnstr, rodType, nSegs=nSegs, isRod=1, outs=outs, rod_attachment=attachment) )
                                
                                if ("body" in attachment) or ("turbine" in attachment):
                                    # attach to body here
                                    BodyID = int("".join(filter(str.isdigit, attachment)))
                                    if len(self.bodyList) < BodyID:
                                        self.bodyList.append( Body(self, 1, 0, np.zeros(6)))
                                        
                                    self.bodyList[BodyID-1].attachRod(num, np.hstack([rA,rB]))
                                    
                                else: # (in progress - unsure if htis works) <<<
                                    self.rodList[-1].rA = rA #.setEndPosition(rA, 0)  # set initial end A position
                                    self.rodList[-1].rB = rB #.setEndPosition(rB, 1)  # set initial end B position
                                
                            line = next(f)
                            
                    
                    # get properties of each Point
                    if line.count('---') > 0 and (line.upper().count('POINTS') > 0 or line.upper().count('POINT LIST') > 0 or line.upper().count('POINT PROPERTIES') > 0 or line.upper().count('CONNECTION PROPERTIES') > 0 or line.upper().count('NODE PROPERTIES') > 0):
                        line = next(f) # skip this header line, plus channel names and units lines
                        line = next(f)
                        line = next(f)
                        while line.count('---') == 0:
                            entries = line.split()         # entries:  ID   Attachment  X       Y     Z      Mass   Volume  CdA    Ca
                            entry0 = entries[0].lower()          
                            entry1 = entries[1].lower() 
                            
                            
                            num = np.int_("".join(c for c in entry0 if not c.isalpha()))  # remove alpha characters to identify Point #
                            
                            
                            if ("anch" in entry1) or ("fix" in entry1):
                                pointType = 1
                                # attach to ground body for ease of identifying anchors
                                self.groundBody.attachPoint(num,entries[2:5]) 
                                
                            elif ("body" in entry1) or ("turbine" in entry1):
                                pointType = 1
                                # attach to body here
                                BodyID = int("".join(filter(str.isdigit, entry1)))
                                if len(self.bodyList) < BodyID:
                                    self.bodyList.append( Body(self, 1, 0, np.zeros(6)))
                                    print("New body added")  # <<< should add consistent warnings in these cases
                                
                                rRel = np.array(entries[2:5], dtype=float)
                                self.bodyList[BodyID-1].attachPoint(num, rRel)
                                
                            elif ("fair" in entry1) or ("ves" in entry1) or ("couple" in entry1):
                                # for coupled point type, just set it up that same way in MoorPy (attachment to a body not needed, right?)
                                pointType = -1                            
                                '''
                                # attach to a generic platform body (and make it if it doesn't exist)
                                if len(self.bodyList) > 1:
                                    raise ValueError("Generic Fairlead/Vessel-type points aren't supported when multiple bodies are defined.")
                                if len(self.bodyList) == 0:
                                    #print("Adding a body to attach fairlead points to.")
                                    self.bodyList.append( Body(self, 1, 0, np.zeros(6)))#, m=m, v=v, rCG=rCG) )
                                
                                rRel = np.array(entries[2:5], dtype=float)
                                self.bodyList[0].attachPoint(num, rRel)    
                                '''
                                    
                            elif ("con" in entry1) or ("free" in entry1):
                                pointType = 0
                            else:
                                print("Point type not recognized")
                            
                            if 'seabed' in entries[4]:
                                entries[4] = -self.depth
                            r = np.array(entries[2:5], dtype=float)
                            m  = float(entries[5])
                            v  = float(entries[6])
                            CdA= float(entries[7])
                            Ca = float(entries[8])
                            self.pointList.append( Point(self, num, pointType, r, m=m, v=v, CdA=CdA, Ca=Ca) )
                            line = next(f)
                            
                            
                    # get properties of each line
                    if line.count('---') > 0 and (line.upper().count('LINES') > 0 or line.upper().count('LINE LIST') > 0 or line.upper().count('LINE PROPERTIES') > 0):
                        line = next(f) # skip this header line, plus channel names and units lines
                        line = next(f)
                        line = next(f)
                        while line.count('---') == 0:
                            entries = line.split()  # entries: ID  LineType  AttachA  AttachB  UnstrLen  NumSegs   Outputs
                                                    
                            num    = np.int_(entries[0])
                            lUnstr = np.float_(entries[4])
                            lineType = self.lineTypes[entries[1]]
                            nSegs  = np.int_(entries[5])         
                            outs = entries[6]
                            
                            #lineList.append( Line(dirName, num, lUnstr, dia, nSegs) )
                            self.lineList.append( Line(self, num, lUnstr, lineType, nSegs=nSegs, outs=outs)) #attachments = [int(entries[4]), int(entries[5])]) )
                            
                            # attach end A
                            numA = int("".join(filter(str.isdigit, entries[2])))  # get number from the attachA string
                            if entries[2][0] in ['r','R']:    # if id starts with an "R" or "Rod"  
                                if numA <= len(self.rodList) and numA > 0:
                                    if entries[2][-1] in ['a','A']:
                                        self.rodList[numA-1].attachLine(num, 0)  # add line (end A, denoted by 0) to rod >>end A, denoted by 0<<
                                        self.lineList[num-1].lr_attachment[0] = 'a'
                                    elif entries[2][-1] in ['b','B']: 
                                        self.rodList[numA-1].attachLine(num, 0)  # add line (end A, denoted by 0) to rod >>end B, denoted by 1<<
                                        self.lineList[num-1].lr_attachment[0] = 'b'
                                    else:
                                        raise ValueError(f"Rod end (A or B) must be specified for line {num} end A attachment. Input was: {entries[2]}")
                                else:
                                    raise ValueError(f"Rod ID ({numA}) out of bounds for line {num} end A attachment.") 
                            
                            else:     # if J starts with a "C" or "Con" or goes straight ot the number then it's attached to a Connection
                                if numA <= len(self.pointList) and numA > 0:  
                                    self.pointList[numA-1].attachLine(num, 0)  # add line (end A, denoted by 0) to Point
                                else:
                                    raise ValueError(f"Point ID ({numA}) out of bounds for line {num} end A attachment.") 

                            # attach end B
                            numB = int("".join(filter(str.isdigit, entries[3])))  # get number from the attachA string
                            if entries[3][0] in ['r','R']:    # if id starts with an "R" or "Rod"  
                                if numB <= len(self.rodList) and numB > 0:
                                    if entries[3][-1] in ['a','A']:
                                        self.rodList[numB-1].attachLine(num, 1)  # add line (end B, denoted by 1) to rod >>end A, denoted by 0<<
                                        self.lineList[num-1].lr_attachment[1] = 'a'
                                    elif entries[3][-1] in ['b','B']: 
                                        self.rodList[numB-1].attachLine(num, 1)  # add line (end B, denoted by 1) to rod >>end B, denoted by 1<<
                                        self.lineList[num-1].lr_attachment[1] = 'b'
                                    else:
                                        raise ValueError(f"Rod end (A or B) must be specified for line {num} end B attachment. Input was: {entries[2]}")
                                else:
                                    raise ValueError(f"Rod ID ({numB}) out of bounds for line {num} end B attachment.") 
                            
                            else:     # if J starts with a "C" or "Con" or goes straight ot the number then it's attached to a Connection
                                if numB <= len(self.pointList) and numB > 0:  
                                    self.pointList[numB-1].attachLine(num, 1)  # add line (end B, denoted by 1) to Point
                                else:
                                    raise ValueError(f"Point ID ({numB}) out of bounds for line {num} end B attachment.") 

                            line = next(f)  # advance to the next line

                    # get options entries
                    if line.count('---') > 0 and "options" in line.lower():
                        #print("READING OPTIONS")
                        line = next(f) # skip this header line
                        
                        while line.count('---') == 0:
                            entries = line.split()       
                            entry0 = entries[0].lower() 
                            entry1 = entries[1].lower() 
                                                    
                            # grab any parameters used by MoorPy
                            if entry1 == "g" or entry1 == "gravity":
                                self.g  = float(entries[0])
                                
                            elif entry1 == "wtrdepth" or entry1 == "depth" or entry1 == "wtrdpth":
                                try:
                                    self.depth = float(entries[0])
                                except:
                                    self.depth = 0.0
                                    print("Warning: non-numeric depth in input file - MoorPy will ignore it.")
                                
                            elif entry1=="rho" or entry1=="wtrdnsty":
                                self.rho = float(entries[0])
                            
                            # also store a dict of all parameters that can be regurgitated during an unload
                            self.MDoptions[entries[1]] = entries[0]
                            
                            line = next(f)
                    if line.count('---') > 0 and "output" in line.lower():
                        #print("READING OUTPUTS")
                        line = next(f) # skip this header line
                        
                        while line.count('---') == 0:
             
                            
                            # also store a dict of all parameters that can be regurgitated during an unload
                            self.MDoutputs.append(line.strip())
                            line = next(f)

                f.close()  # close data file
            # any error check? <<<
            
            print(f"Mooring input file '{filename}' loaded successfully.")

    def unload(self, fileName, MDversion=2, line_dL=0, rod_dL=0, Lm = 0):
            '''Unloads a MoorPy system into a MoorDyn-style input file

            Parameters
            ----------
            fileName : string
                file name of output file to hold MoorPy System.
            line_dL : float, optional
                Optional specified for target segment length when discretizing Lines
            rod_dL : float, optional
                Optional specified for target segment length when discretizing Rods
            outputList : list of strings, optional
                Optional list of additional requested output channels
            Lm : float
                Mean load on mooring line as FRACTION of MBL, used for dynamic stiffness calculation. Only used if line type has a nonzero EAd

            Returns
            -------
            None.

            '''
            if MDversion==1:
                #For version MoorDyn v1

                #Collection of default values, each can be customized when the method is called
                
                # Set up the dictionary that will be used to write the OPTIONS section
                MDoptionsDict = dict(dtM=0.001, kb=3.0e6, cb=3.0e5, TmaxIC=60)        # start by setting some key default values
                # Other available options: Echo=False, dtIC=2, CdScaleIC=10, threshIC=0.01
                MDoptionsDict.update(self.MDoptions)                                  # update the dict with any settings saved from an input file
                MDoptionsDict.update(dict(g=self.g, WtrDepth=self.depth, rho=self.rho))  # lastly, apply any settings used by MoorPy
                MDoptionsDict.update(dict(WriteUnits=0))    # need this for WEC-Sim

                # Some default settings to fill in if coefficients aren't set
                #lineTypeDefaults = dict(BA=-1.0, EI=0.0, Cd=1.2, Ca=1.0, CdAx=0.2, CaAx=0.0)
                lineTypeDefaults = dict(BA=-1.0, cIntDamp=-0.8, EI=0.0, Can=1.0, Cat=1.0, Cdn=1.0, Cdt=0.5)
                rodTypeDefaults  = dict(Cd=1.2, Ca=1.0, CdEnd=1.0, CaEnd=1.0)
                
                # bodyDefaults = dict(IX=0, IY=0, IZ=0, CdA_xyz=[0,0,0], Ca_xyz=[0,0,0])
                
                # Figure out mooring line attachments (Create a ix2 array of connection points from a list of m points)
                connection_points = np.empty([len(self.lineList),2])                   #First column is Anchor Node, second is Fairlead node
                for point_ind,point in enumerate(self.pointList,start = 1):                    #Loop through all the points
                    for (line,line_pos) in zip(point.attached,point.attachedEndB):          #Loop through all the lines #s connected to this point
                        if line_pos == 0:                                                       #If the A side of this line is connected to the point
                            connection_points[line -1,0] = point_ind                                #Save as as an Anchor Node
                            #connection_points[line -1,0] = self.pointList.index(point) + 1
                        elif line_pos == 1:                                                     #If the B side of this line is connected to the point
                            connection_points[line -1,1] = point_ind                                #Save as a Fairlead node
                            #connection_points[line -1,1] = self.pointList.index(point) + 1
                
                #Outputs List
                Outputs = [f"FairTen{i+1}" for i in range(len(self.lineList))]        # for now, have a fairlead tension output for each line
                #Outputs.append("Con2Fz","Con3Fz","Con6Fz","Con7Fz","Con10Fz","Con11Fz","L3N20T","L6N20T","L9N20T")
    
                
    
                print('attempting to write '+fileName +' for MoorDyn v'+str(MDversion))
                #Array to add strings to for each line of moordyn input file
                L = []                   
                
                
                # Generate text for the MoorDyn input file
                L.append('Mooring line data file for MoorDyn in Lines.dll')
                #L.append(f"MoorDyn v{MDversion} Input File ")
                #L.append("Generated by MoorPy")
                #L.append("{:5}    Echo      - echo the input file data (flag)".format(str(Echo).upper()))
                    
                
                #L.append("---------------------- LINE TYPES -----------------------------------------------------")
                L.append("---------------------- LINE DICTIONARY -----------------------------------------------------")
                #L.append(f"{len(self.lineTypes)}    NTypes   - number of LineTypes")
                #L.append("LineType         Diam     MassDen   EA        cIntDamp     EI     Can    Cat    Cdn    Cdt")
                #L.append("   (-)           (m)      (kg/m)    (N)        (Pa-s)    (N-m^2)  (-)    (-)    (-)    (-)")
                L.append("LineType         Diam     MassDenInAir   EA        BA/-zeta     Can    Cat    Cdn    Cdt")
                L.append("   (-)           (m)        (kg/m)       (N)       (Pa-s/-)     (-)    (-)    (-)    (-)")

                for key, lineType in self.lineTypes.items(): 
                    di = lineTypeDefaults.copy()  # start with a new dictionary of just the defaults
                    di.update(lineType)           # then copy in the lineType's existing values
                    #L.append("{:<12} {:7.4f} {:8.6f}  {:7.3e} {:7.3e} {:7.3e}   {:<7.3f} {:<7.3f} {:<7.2f} {:<7.2f}".format(
                            #key, di['d_vol'], di['m'], di['EA'], di['cIntDamp'], di['EI'], di['Can'], di['Cat'], di['Cdn'], di['Cdt']))
                    L.append("{:<12} {:7.4f} {:8.6f}  {:7.3e} {:7.3e}       {:<7.3f} {:<7.3f} {:<7.2f} {:<7.2f}".format(
                            # key, di['d_vol'], di['m'], di['EA'], di['BA'], di['Can'], di['Cat'], di['Cdn'], di['Cdt']))
                            key, di['d_vol'], di['m'], di['EA'], di['BA'], di['Ca'], di['CaAx'], di['Cd'], di['CdAx']))
                
                #L.append("---------------------- POINTS ---------------------------------------------------------")
                L.append("---------------------- NODE PROPERTIES ---------------------------------------------------------")
                #L.append(f"{len(self.pointList)}    NConnects   - number of connections including anchors and fairleads")
                L.append("Node    Type         X        Y        Z        M      V      FX     FY     FZ    CdA    CA ")
                L.append("(-)     (-)         (m)      (m)      (m)      (kg)   (m^3)  (kN)   (kN)   (kN)   (m^2)  (-)")
                #L.append("ID  Attachment     X       Y       Z          Mass   Volume  CdA    Ca")
                #L.append("(#)   (-)         (m)     (m)     (m)         (kg)   (m^3)  (m^2)   (-)")
                
                for point in self.pointList:
                    point_pos = point.r             # get point position in global reference frame to start with
                    if point.type == 1:             # point is fixed or attached (anch, body, fix)
                        point_type = 'Fixed'
                        
                        #Check if the point is attached to body
                        for body in self.bodyList:
                            for attached_Point in body.attachedP:
                                
                                if attached_Point == point.number:
                                    #point_type = "Body" + str(body.number)
                                    point_type = "Vessel"
                                    point_pos = body.rPointRel[body.attachedP.index(attached_Point)]   # get point position in the body reference frame
                        
                    elif point.type == 0:           # point is coupled externally (con, free)
                        point_type = 'Connect'
                            
                    elif point.type == -1:          # point is free to move (fair, ves)
                        point_type = 'Vessel'
                    
                    L.append("{:<4d} {:9} {:8.6f} {:8.6f} {:8.6f} {:9.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(
                            point.number,point_type, point_pos[0],point_pos[1],point_pos[2], point.m, point.v, point.fExt[0],point.fExt[1],point.fExt[2], point.CdA, point.Ca))
                    
                
                #L.append("---------------------- LINES -----------------------------------------------------")
                L.append("---------------------- LINE PROPERTIES -----------------------------------------------------")
                #L.append(f"{len(self.lineList)}    NLines   - number of line objects")
                #L.append("Line      LineType   UnstrLen  NumSegs  AttachA  AttachB  Outputs")
                #L.append("(-)         (-)       (m)        (-)     (-)      (-)     (-)")
                #L.append("ID    LineType      AttachA  AttachB  UnstrLen  NumSegs  LineOutputs")
                #L.append("(#)    (name)        (#)      (#)       (m)       (-)     (-)")
                L.append("Line      LineType   UnstrLen  NumSegs  NodeAnch  NodeFair  Flags/Outputs")
                L.append("(-)         (-)       (m)        (-)      (-)       (-)         (-)")

                for i,line in enumerate(self.lineList):
                    L.append("{:<4d} {:<15} {:8.3f} {:5d} {:7d} {:8d}      {}"
                            .format(line.number, line.type['name'], line.L, line.nNodes-1, int(connection_points[i,0]), int(connection_points[i,1]), line.outs))
                
                
                #L.append("---------------------- OPTIONS ----------------------------------------")
                L.append("---------------------- SOLVER OPTIONS ----------------------------------------")

                for key, val in MDoptionsDict.items():
                    L.append(f"{val:<15}  {key}")
                
                """
                #Solver Options
                L.append("{:<9.3f}dtM          - time step to use in mooring integration (s)".format(float(dtm)))
                L.append("{:<9.0e}kbot           - bottom stiffness (Pa/m)".format(kbot))
                L.append("{:<9.0e}cbot           - bottom damping (Pa-s/m)".format(cbot))
                L.append("{:<9.0f}dtIC      - time interval for analyzing convergence during IC gen (s)".format(int(dtIC)))
                L.append("{:<9.0f}TmaxIC      - max time for ic gen (s)".format(int(TmaxIC)))
                L.append("{:<9.0f}CdScaleIC      - factor by which to scale drag coefficients during dynamic relaxation (-)".format(int(CdScaleIC)))
                L.append("{:<9.2f}threshIC      - threshold for IC convergence (-)".format(threshIC))
                
                #Failure Header
                """
                
                L.append("--------------------------- OUTPUTS --------------------------------------------")
                                
                for Output in self.MDoutputs:
                    L.append(Output)
                #L.append("END")
                    
                    
                L.append('--------------------- need this line ------------------')
                
                
                #Write the text file
                with open(fileName, 'w') as out:
                    for x in range(len(L)):
                        out.write(L[x])
                        out.write('\n')
                
                print('Successfully written '+fileName +' input file using MoorDyn v1')
            
            
            
            elif MDversion==2:
                #For version MoorDyn v?.??
                
                #Collection of default values, each can be customized when the method is called
                
                #Header
                #version = 
                #description = 
                
                # Set up the dictionary that will be used to write the OPTIONS section
                MDoptionsDict = dict(dtM=0.001, kb=3.0e6, cb=3.0e5, TmaxIC=60)        # start by setting some key default values
                MDoptionsDict.update(self.MDoptions)                                  # update the dict with any settings saved from an input file
                MDoptionsDict.update(dict(g=self.g, depth=self.depth, rho=self.rho))  # lastly, apply any settings used by MoorPy
                
                # Some default settings to fill in if coefficients aren't set
                lineTypeDefaults = dict(BA=-1.0, EI=0.0, Cd=1.2, Ca=1.0, CdAx=0.2, CaAx=0.0)
                rodTypeDefaults  = dict(Cd=1.2, Ca=1.0, CdEnd=1.0, CaEnd=1.0)
                
                # Figure out mooring line attachments (Create a ix2 array of connection points from a list of m points)
                connection_points = np.empty([len(self.lineList),2], dtype='<U5')                   #First column is Anchor Node, second is Fairlead node
                for point_ind,point in enumerate(self.pointList,start = 1):                    #Loop through all the points
                    for (line,line_pos) in zip(point.attached,point.attachedEndB):          #Loop through all the lines #s connected to this point
                        if line_pos == 0:                                                       #If the A side of this line is connected to the point
                            connection_points[line -1,0] = str(point_ind)                                #Save as as an Anchor Node
                            #connection_points[line -1,0] = self.pointList.index(point) + 1
                        elif line_pos == 1:                                                     #If the B side of this line is connected to the point
                            connection_points[line -1,1] = str(point_ind)                                #Save as a Fairlead node
                            #connection_points[line -1,1] = self.pointList.index(point) + 1     
                for rod_ind,rod in enumerate(self.rodList,start = 1): #Loop through all the rods                         
                    for (line,line_pos) in zip(rod.attached,rod.attachedEndB):          #Loop through all the lines #s connected to this rod
                        if line_pos == 0:                                                      #If the A side of this line is connected to the rod
                            connection_points[line -1,0] = ('r'+str(rod_ind)+self.lineList[line-1].lr_attachment[0])                                 #Save as as an Anchor Node
                        elif line_pos == 1:                                                     #If the B side of this line is connected to the rod
                            connection_points[line -1,1] = ('r'+str(rod_ind)+self.lineList[line-1].lr_attachment[1])                                 #Save as a Fairlead node

                print('attempting to write '+fileName +' for MoorDyn v'+str(MDversion))
                #Array to add strings to for each line of moordyn input file
                L = []                   
                
                
                # Generate text for the MoorDyn input file 
                
                L.append(f"MoorDyn v{MDversion} Input File ")
                #if "description" in locals():
                    #L.append("MoorDyn input for " + description)
                #else: 
                L.append("Generated by MoorPy")
                    
                    
                L.append("---------------------- LINE TYPES --------------------------------------------------")
                L.append("TypeName      Diam     Mass/m     EA     BA/-zeta     EI        Cd      Ca      CdAx    CaAx")
                L.append("(name)        (m)      (kg/m)     (N)    (N-s/-)    (N-m^2)     (-)     (-)     (-)     (-)")
                
                for key, lineType in self.lineTypes.items(): 
                    di = lineTypeDefaults.copy()  # start with a new dictionary of just the defaults
                    di.update(lineType)           # then copy in the lineType's existing values
                    if 'EAd' in di.keys() and di['EAd'] > 0:
                        if Lm > 0:
                            print('Calculating dynamic stiffness with Lm = ' + str(Lm)+'* MBL')
                            L.append("{:<12} {:7.4f} {:8.6f}  {:7.3e}|{:7.3e} 4E9|11e6 {:7.3e}   {:<7.3f} {:<7.3f} {:<7.2f} {:<7.2f}".format(
                                key, di['d_vol'], di['m'], di['EA'], di['EAd'] + di['EAd_Lm']*Lm*di['MBL'], di['EI'], di['Cd'], di['Ca'], di['CdAx'], di['CaAx']))
                        else:
                            print('No mean load provided!!! using the static EA value ONLY')
                            L.append("{:<12} {:7.4f} {:8.6f}  {:7.3e} {:7.3e} {:7.3e}   {:<7.3f} {:<7.3f} {:<7.2f} {:<7.2f}".format(
                                    key, di['d_vol'], di['m'], di['EA'], di['BA'], di['EI'], di['Cd'], di['Ca'], di['CdAx'], di['CaAx']))
                    else:
                        L.append("{:<12} {:7.4f} {:8.6f}  {:7.3e} {:7.3e} {:7.3e}   {:<7.3f} {:<7.3f} {:<7.2f} {:<7.2f}".format(
                                key, di['d_vol'], di['m'], di['EA'], di['BA'], di['EI'], di['Cd'], di['Ca'], di['CdAx'], di['CaAx']))
                
                
                L.append("--------------------- ROD TYPES -----------------------------------------------------")
                L.append("TypeName      Diam     Mass/m    Cd     Ca      CdEnd    CaEnd")
                L.append("(name)        (m)      (kg/m)    (-)    (-)     (-)      (-)")
                
                for key, rodType in self.rodTypes.items(): 
                    di = rodTypeDefaults.copy()
                    di.update(rodType)
                    L.append("{:<15} {:7.4f} {:8.6f} {:<7.3f} {:<7.3f} {:<7.3f} {:<7.3f}".format(
                            key, di['d_vol'], di['m'], di['Cd'], di['Ca'], di['CdEnd'], di['CaEnd']))
                
                
                L.append("----------------------- BODIES ------------------------------------------------------")
                L.append("ID   Attachment    X0     Y0     Z0     r0      p0     y0     Mass          CG*          I*      Volume   CdA*   Ca*")
                L.append("(#)     (-)        (m)    (m)    (m)   (deg)   (deg)  (deg)   (kg)          (m)         (kg-m^2)  (m^3)   (m^2)  (-)")
                
                for body in self.bodyList:
                    attach = ['coupled','free','fixed'][[-1,0,1].index(body.type)]                      # pick correct string based on body type
                    L.append("{:<4d}  {:10}  {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f} ".format(
                            body.number, attach, body.r6[0],body.r6[1],body.r6[2],np.rad2deg(body.r6[3]),np.rad2deg(body.r6[4]),np.rad2deg(body.r6[5])
                            )+ "{:<9.4e}  {:.2f}|{:.2f}|{:.2f} {:9.3e} {:6.2f} {:6.2f} {:5.2f}".format(
                            body.m, body.rCG[0],body.rCG[1],body.rCG[2], body.I[0], body.v, body.CdA[0], body.Ca[0]))
                            
                            # below is a more thorough approach to see about in future
                            #)+ "{:<9.2f}  {:<5.2f}|{:<5.2f}|{:<5.2f}  {:<5.2f}|{:<5.2f}|{:<5.2f}  {:<5.2f}  {:<5.2f}|{:<5.2f}|{:<5.2f}  {:<5.2f}|{:<5.2f}|{:<5.2f}".format(
                            #body.m, body.rCG[0],body.rCG[1],body.rCG[2], body.I[0],body.I[1],body.I[2],
                            #body.v, body.CdA[0],body.CdA[1],body.CdA[2], body.Ca[0],body.Ca[1],body.Ca[2]))
                            
                L.append("---------------------- RODS ---------------------------------------------------------")
                L.append("ID   RodType  Attachment  Xa    Ya    Za    Xb    Yb    Zb   NumSegs  RodOutputs")
                L.append("(#)  (name)    (#/key)    (m)   (m)   (m)   (m)   (m)   (m)  (-)       (-)")
                
                for rod in self.rodList:
                    try:
                        nSegs = int(np.ceil(rod.L/line_dL)) if line_dL>0 else rod.nNodes-1  # if target dL given, set nSegs based on it instead of line.nNodes
                    except:
                        nSegs = 0
                        print("Rod number "+str(rod.number)+" nSegs set to 0")

                    if nSegs == 0: # rod is held as point object
                        L.append("{:<4d}  {:<15}  {:<7}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}  {:4d}  {:<5}".format(
                            rod.number, rod.type['name'], rod.rod_attachment, rod.r[0], rod.r[1], rod.r[2], rod.r[0], rod.r[1], rod.r[2], nSegs, rod.outs))                
                    else:
                        L.append("{:<4d}  {:<15}  {:<7}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}  {:4d}  {:<5}".format(
                            rod.number, rod.type['name'], rod.rod_attachment, rod.rA[0], rod.rA[1], rod.rA[2], rod.rB[0], rod.rB[1], rod.rB[2], nSegs, rod.outs))                
                
                L.append("---------------------- POINTS -------------------------------------------------------")
                L.append("ID  Attachment     X       Y       Z           Mass  Volume  CdA    Ca")
                L.append("(#)   (-)         (m)     (m)     (m)          (kg)  (mˆ3)  (m^2)   (-)")
                
                for point in self.pointList:
                    point_pos = point.r             # get point position in global reference frame to start with
                    if point.type == 1:             # point is fixed or attached (anch, body, fix)
                        point_type = 'Fixed'
                        
                        #Check if the point is attached to body
                        for body in self.bodyList:
                            for attached_Point in body.attachedP:
                                if attached_Point == point.number:
                                    point_type = "Body" + str(body.number)
                                    point_pos = body.rPointRel[body.attachedP.index(attached_Point)]   # get point position in the body reference frame
                        
                    elif point.type == 0:           # point is coupled externally (con, free)
                        point_type = 'Free'
                            
                    elif point.type == -1:          # point is free to move (fair, ves)
                        point_type = 'Coupled'
                    
                    L.append("{:<4d} {:9} {:8.6f} {:8.6f} {:8.6f} {:9.3f} {:6.3f} {:6.3f} {:6.3f}".format(
                            point.number,point_type, point_pos[0],point_pos[1],point_pos[2], point.m, point.v, point.CdA, point.Ca))
                    
                
                L.append("---------------------- LINES --------------------------------------------------------")
                L.append("ID    LineType      AttachA  AttachB  UnstrLen  NumSegs  LineOutputs")
                L.append("(#)    (name)        (#)      (#)       (m)       (-)     (-)")
                
                for i,line in enumerate(self.lineList):
                    nSegs = int(np.ceil(line.L/line_dL)) if line_dL>0 else line.nNodes-1  # if target dL given, set nSegs based on it instead of line.nNodes
                
                    L.append("{:<4d} {:<15} {:^5}   {:^5}   {:8.3f}   {:4d}       {:<5}".format(
                            line.number, line.type['name'], connection_points[i,0], connection_points[i,1], line.L, nSegs, line.outs))
                
                
                L.append("---------------------- OPTIONS ------------------------------------------------------")


                for key, val in MDoptionsDict.items():
                    if 'writelog' in key.lower(): # make sure writelog is top of the list
                        L.append(f"{val:<15}  {key}")

                for key, val in MDoptionsDict.items():
                    if 'writelog' not in key.lower():
                        L.append(f"{val:<15}  {key}")
                
                
                #Failure Header
                #Failure Table
                
                
                L.append("----------------------- OUTPUTS -----------------------------------------------------")
                for Output in self.MDoutputs:
                    L.append(Output)
                L.append("END")
                    
                    
                L.append('--------------------- need this line ------------------------------------------------')
                
                
                #Write the text file
                with open(fileName, 'w') as out:
                    for x in range(len(L)):
                        out.write(L[x])
                        out.write('\n')
            
                print('Successfully written '+fileName +' input file using MoorDyn v2')