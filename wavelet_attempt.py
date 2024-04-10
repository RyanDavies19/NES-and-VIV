# This is not currently working, an attempt to convert the wavelet script from matlab to python

import moorpy as mp
import matplotlib.pyplot as plt
import numpy as np
import math

def nextpow2(N):
    # Calculate log2 of N
    a = int(math.log2(N))
 
    # If 2^a is equal to N, return N
    if 2**a == N:
        return N
     
    # Return 2^(a + 1)
    return 2**(a + 1)

def freq_inst_morlet(xnew,FS,fi,ff,nf,Fo):
   # ###################################################
   # Wavelet Transform using Morlet Wavelet
   # ##################################################
   #
   # Inputs
   # ------
   # xnew = signal with dimensions N x 1
   # Fs = sampling frequency
   # fi = low frequency bound
   # ff = high frequency bound
   # nf = number of frequency points (higher is results in finer plots, 100
   # or 200 are good starting points)
   # Fo = Mother wavelet frequency (2 or 4 is best)
   #
   # Outputs
   # -------
   # t0 = Time vector based on length of xnew and FS
   # interval_freq = Frequency vector
   # module = Absolute value of wavelet transform of xnew

   ## Transform Parameters
   dt=1/FS 
   l_x=len(xnew) 
   t0=np.arange(0,l_x*dt-dt, dt) 
   df = (ff-fi)/nf 
   interval_freq=np.arange(fi,ff,df) 
   a = Fo/interval_freq 

   ## FFT Parameters
   ntemps = len(t0) 
   nfourier = nextpow2(ntemps) 
   npt = int(nfourier/2)
   freq = 1/dt*np.arange(0,npt-1)/nfourier  # Frequency vector

   ## Compute FFT of xnew
   tff_full = np.fft.fft(xnew,nfourier) 
   tff= tff_full[npt+1:] # Only select the front half of the array

   # freq = np.array([1,2,3,4])
   # a = np.array([5,6,7])
   # tff = np.array([8,9,10,11])
   ## Vectorized Computation of Wavelet Transform of xnew
   temp1 = np.matmul(freq.reshape(len(freq),1),a.reshape(1,len(a)))
   temp2 = (2**0.5)*np.exp(-0.5*(2*np.pi*(temp1-Fo))**2) * np.sqrt(a)
   noyau2 = np.conjugate(temp2) * tff.reshape(len(tff),1)
   # for i in range(len(freq)):
   #    if freq[i] == 0.0:
   #       noyau2[i,:] = 0
   resut2 = np.fft.ifft(noyau2,nfourier, axis = 0) # TODO: this is what is taking a ton of time
   resut3 = resut2[0:ntemps,:] 
   module  = abs(resut3) 

   return t0,interval_freq,module

if __name__ == "__main__":

    # CONTOUR PLOTS
    data, ch, channels, units = mp.read_mooring_file('', 'catenary_riser_smallC_Line1_copy.out') # remember number starts on 1 rather than 0
    time = data[:,ch['Time']]
    # segs = np.arange(1,51,1)
    nodes = np.arange(0,51,1)
    diam = 0.027 # cable diameter

    # make strain array
    # strain = np.zeros((len(time),len(segs)))
    # make py/D array
    pyD = np.zeros((len(time),len(nodes)))
    # make lift force array
    # fviv = np.zeros((len(time),len(nodes)))

    i = 0
    j = 0 
    k = 0
    for channel in channels:
        # if 'St' in channel:
        #    strain[:,i] = data[:,ch[channel]]
        #    i+=1
        if 'py' in channel:
            pyD[:,j] = data[:,ch[channel]]/diam
            j+=1
        # if 'Vy' in channel:
        #    fviv[:,k] = data[:,ch[channel]]
        #    k+=1
    del data, ch, channels, units
    # transpose to get x and y in right place
    # strain = strain.transpose()
    pyD = pyD.transpose()
    # fviv = fviv.transpose()

    # # plot conplots

    # ### strain
    # strain = np.flip(strain,0)
    # fig1, ax1 = plt.subplots()
    # stcon = ax1.imshow(strain, extent = [time.min(),time.max(),segs.min(),segs.max()], cmap = 'jet', aspect = 'auto')
    # cbar1 = fig1.colorbar(stcon)
    # fig1.suptitle('Segment Strain, 0.2 m/s current')
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Segment number')
    # cbar1.set_label('Strain (-)')

    ### py/D
    pyD = np.flip(pyD,0)
    # fig2, ax2 = plt.subplots()
    # ycon = ax2.imshow(pyD[:,int(18/0.001):int(30/0.001)], extent = [18,30,nodes.min(),nodes.max()], cmap = 'jet', aspect = 'auto')
    # cbar2 = fig2.colorbar(ycon)
    # fig2.suptitle('Y displacement normalized by diameter, 0.2 m/s current')
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Node number')
    # cbar2.set_label('y displacement / riser diameter (-)')

    # ### VIV force
    # fviv = np.flip(fviv,0)
    # fig3, ax3 = plt.subplots()
    # vcon = ax3.imshow(fviv, extent = [time.min(),time.max(),nodes.min(),nodes.max()], cmap = 'jet', aspect = 'auto')
    # cbar3 = fig3.colorbar(vcon)
    # fig3.suptitle('Lift Force in Y direction')

    # plt.show()

    ## Compute Wavelet Transform
    start_index = 1 
    tmax = 60 
    dtout = 0.001 
    end_index = (tmax / dtout) 

    # The data files you import with nes and without    
    signal  = pyD[17,:] 
    T_span = time

    #skip filtering
    # Design a high-pass filter with a very low cutoff frequency
    #  hpFilt = designfilt('highpass', 'FilterOrder', 1, 'HalfPowerFrequency', 0.1, 'SampleRate', Fs) 
    # # Apply the high-pass filter
    # signal  = filtfilt(hpFilt, signal ) 
    # z10_full = filtfilt(hpFilt, z10_full) 

    # Tspan_full = (0:0.1:100)' 
    Fs = 1/dtout 
    freq_lb = 0.0 
    freq_up = 40.0 
    Fo = 1 
    nf = 800  # number of frequency points (higher is results in finer plots, 100 or 200 are good starting points)
        
    t_0, freq, mod_signal  = freq_inst_morlet(signal , Fs, freq_lb, freq_up, nf, Fo) 

    # Normalize Wavelet Transform Amplitude for Plotting
    # mod_R0 = mod_R0'  # R0
    # mod2plot_R0 = mod_R0/max(max(mod_R0)) 
    # mod = mod'  # L1
    mod2plot_signal  = mod_signal  
    # mod2plot_y1 = mod_y1/max(max(mod_y1)) 
    # mod2plot_signal  = mod_signal /max(max(mod_signal )) 

    ## Ploting Wavelet Transform
    fig4, ax4 = plt.subplots()
    wavelet = ax4.imshow(mod2plot_signal**1.5 , extent = [T_span.min(), T_span.max(), freq_lb, freq_up], cmap = 'jet', aspect = 'auto') 
    cbar4 = fig4.colorbar(wavelet)
    ax4.set_xlabel('Time (s)') 
    ax4.set_ylabel('Frequency (Hz)') 
    fig4.suptitle('<insert title>')

    plt.show()