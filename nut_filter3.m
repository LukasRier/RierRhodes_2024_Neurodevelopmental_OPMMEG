function data = nut_filter3(data,filtclass,filttype,filtorder,lofreq,hifreq,fs,baseline)
% Used to filter time series -- should replace nut_filter eventually
% TODO: need to adjust parameters in nut_activation_viewer & nut_timef_viewer
% 
% nut_filter2(data,filtclass,filttype,filtorder,lofreq,hifreq,fs,baseline)
% data = [samples x channels] or [samples x channels x trials]
% filtclass = 'butter', 'cheby1', 'cheby2', 'firpm', 'firls'
% filttype = 'notch', 'bp' (bandpass), or 'baseline_only'
% filtorder = filter order
% lofreq = low cutoff (Hz)  (0 for lowpass)
% hifreq = high cutoff (Hz) (inf for highpass)
% fs = sampling rate (Hz)
% baseline = 0 (off) or 1 (subtract mean of whole interval)
%            or, e.g., [1 100] (subtract mean of first 100 points)
%
% Program is called by nut_activation_viewer.m and nut_beamforming_gui.m

if (strcmp(filttype,'baseline_only') & ( (length(baseline)==1 & baseline==0) | isempty(baseline)))
    error('Crackhead! you have baseline_only selected as filttype, but baseline is not set right')
end

% apply activation_baseline correction
if(~isempty(baseline))
    if length(baseline) == 2
        f = baseline(1):baseline(2);
    else
        f = 1:size(data,1);
    end
    
    meandata=zeros(size(data));
    meandata = repmat(mean(data(f,:,:),1),size(data,1),1);
    data = data - meandata; % remove mean prior to applying filter (to reduce transients)
    if(baseline~=0)
        clear meandata
    end
end

if hifreq < lofreq
    error('Crackhead! High cutoff frequency is less than low cutoff frequency!');
end

switch(filttype)
    case 'notch'
        [b,a] = filtdef(filtclass,filtorder,lofreq,hifreq,fs,'stop');
    case 'bp'
        [b,a] = filtdef(filtclass,filtorder,lofreq,hifreq,fs);
    case 'baseline_only'
        return
    otherwise
        error('Unknown filter type.')
end


% data forced to be double before going into filtfilt;
% some filter parameters results in NaN or other screwy results with single
for jj=1:size(data,3)
    for ii=1:size(data,2)
        data(:,ii,jj) = filtfilt(b,a,double(data(:,ii,jj)));
    end
end

if(baseline == 0) % no final baseline correction, so add the mean back
    data = data + meandata;
end

function [b,a]=filtdef(filtclass,filtorder,lofreq,hifreq,fs,filttype)
if(~exist('filttype','var'))
    if(isempty(lofreq))
        lofreq = 0;
        filttype='low';
    elseif(isempty(hifreq) || hifreq==inf)
        hifreq = [];
        filttype='high';
    else
        filttype='bandpass';
    end
end
switch(filtclass)
    case 'butter'
        [b,a] = butter(filtorder,2*[lofreq hifreq]/fs);
    case 'cheby1'
        [b,a] = cheby1(filtorder,0.5,2*[lofreq hifreq]/fs,filttype);
    case 'cheby2'
        [b,a] = cheby2(filtorder,20,2*[lofreq hifreq]/fs,filttype);
    case {'firpm','firls'}
        f = 0:0.001:1;
        if rem(length(f),2)~=0
            f(end)=[];
        end
        z = zeros(1,length(f));
        if(isfinite(lofreq))
            [val,pos1] = min(abs(fs*f/2 - lofreq));
        else
            [val,pos2] = min(abs(fs*f/2 - hifreq));
            pos1=pos2;
        end
        if(isfinite(hifreq))
            [val,pos2] = min(abs(fs*f/2 - hifreq));
        else
            pos2 = length(f);
        end
        z(pos1:pos2) = 1;
        a = 1;
        switch(filtclass)
            case 'firpm'
                b = firpm(filtorder,f,z);
            case 'firls'
                b = firls(filtorder,f,z);
        end
end
