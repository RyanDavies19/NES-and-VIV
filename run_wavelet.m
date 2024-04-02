%% Compute Wavelet Transform 32x line1 einai hz3 line2 einai hz4 line3 einai 2hz combo
    start_index = 1;
    end_index = 4*10000-1;
    
% The data files you import with nes and without    
y10_full = verticalnoclearancecopyLine13(start_index:end_index,2+3*5);
z10_full = verticalnoclearancecopyLine2(start_index:end_index,2+3*5);
T_span = verticalnoclearancecopyLine2(start_index:end_index,1);

%skip filtering
    % Design a high-pass filter with a very low cutoff frequency
%  hpFilt = designfilt('highpass', 'FilterOrder', 1, 'HalfPowerFrequency', 0.1, 'SampleRate', Fs);
% % Apply the high-pass filter
% y10_full = filtfilt(hpFilt, y10_full);
% z10_full = filtfilt(hpFilt, z10_full);

    figure
    h1 = plot(T_span,y10_full)
    hold on;
    h2 = plot(T_span,z10_full)
    ylabel('Displacement (m)')
    xlabel('Time (s)')
    title('Displacement time-series NES')
    legend([h1 h2],'Cable with NES','Cable without NES','location','southwest')
    set(gca,'fontsize',16)
    % Tspan_full = (0:0.1:100)';
    Fs = 1./mean(diff(T_span));
    freq_lb = 0;0;0.95/(2*pi);
    freq_up = 40;20;1.15/(2*pi);
    Fo = 1;
    nf = 800; % number of frequency points (higher is results in finer plots, 100
    % or 200 are good starting points)

    [~, freq, mod_y10] = freq_inst_morlet(y10_full, Fs, freq_lb, freq_up, nf, Fo);
    
    % Normalize Wavelet Transform Amplitude for Plotting
    % mod_R0 = mod_R0'; % R0
    % mod2plot_R0 = mod_R0/max(max(mod_R0));
    % mod = mod'; % L1
    mod2plot_y0 = mod_y10;
    % mod2plot_y1 = mod_y1/max(max(mod_y1));
    % mod2plot_y10 = mod_y10/max(max(mod_y10));
    
    
    %% Ploting Wavelet Transform
    figure
    hold on
    h1 = imagesc(T_span, freq, mod2plot_y0'.^1.5);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    set(gca,'YDir','normal');
    clim = get(gca,'clim');
    set(gca,'clim',clim/5);
    shading interp
    cmap_name = flipud(bone.^0.5); % color map
    colorbar;
    colormap(1-gray.^.5)
    set(gca,'fontsize',16)
    ylim([0 40])
    title('Cable with NES Location: {adjust} Depth')
    
[~, freq, mod_y10] = freq_inst_morlet(z10_full, Fs, freq_lb, freq_up, nf, Fo);

    % Normalize Wavelet Transform Amplitude for Plotting
    % mod_R0 = mod_R0'; % R0
    % mod2plot_R0 = mod_R0/max(max(mod_R0));
    % mod = mod'; % L1
    mod2plot_y0 = mod_y10;
    % mod2plot_y1 = mod_y1/max(max(mod_y1));
    % mod2plot_y10 = mod_y10/max(max(mod_y10));
    
    %% Ploting Wavelet Transform
    figure
    h1 = imagesc(T_span, freq, mod2plot_y0'.^1.5);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    set(gca,'YDir','normal');
    clim = get(gca,'clim');
    set(gca,'clim',clim/5);
    shading interp
    cmap_name = flipud(bone.^0.5); % color map
    colorbar;
    colormap(1-gray.^.5)
    set(gca,'fontsize',16)
    caxis([0 5*10^(-4)]); %rescale the color bar
    ylim([0 40])
    title('Cable without NES Location: {adjust} Depth')
