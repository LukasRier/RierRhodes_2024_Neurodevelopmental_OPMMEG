function [burst_amp_D1,burst_amp_D4,burst_amp_D1_std,burst_amp_D4_std] = peak_VE_hmm_burst_amp(sub,ses,project_dir)
restoredefaultpath
% close all
clc

run = 'run-001';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('.\HMM_bursts-master\')) %from https://github.com/ZSeedat/HMM_bursts/

addpath(genpath('D:\GitHub\HMM-MAR-master')) % from https://github.com/OHBA-analysis/HMM-MAR

datadir = [project_dir,'Data',filesep,'BIDS',filesep];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_type = '_task-braille';
filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];
path_main = [datadir,'sub-',sub,filesep,'ses-',ses,filesep];

path_meg_data = [path_main,'meg',filesep];

files_events = [filename,'_events.tsv'];

path_Tstat = [datadir,'derivatives',filesep,'Tstats',filesep,'sub-',sub,filesep]
file_D1_VE = 'D1_peak_VE_1-150_Hz.mat';
file_D4_VE = 'D4_peak_VE_1-150_Hz.mat';
path_hmm = [datadir,'derivatives',filesep,'HMM',filesep,'sub-',sub,filesep];
file_hmm = [filename,'_hmm-outputs.mat'];
file_hmm_amp = [filename,'_hmm-amplitude.mat'];

if ~exist(path_hmm,'dir'); mkdir(path_hmm);end

if ~exist([path_hmm,file_hmm_amp]) |1
    %% Get 1-48 Hz filtered VE timecourse
    S = load([path_Tstat,filesep,file_D1_VE],'VE_index');
    VE_D1 = S.VE_index;
    S = load([path_Tstat,filesep,file_D4_VE],'VE_pinky');
    VE_D4 = S.VE_pinky;
    clear S;


    fs = 1200;
    hp = 5;
    lp = 48;
    [b,a] = butter(4,2*[hp lp]/fs);
    VE_D1_f = [filtfilt(b,a,VE_D1)];
    VE_D4_f = [filtfilt(b,a,VE_D4)];
    VE_D1_f = reshape(VE_D1_f,1,[]);
    VE_D4_f = reshape(VE_D4_f,1,[]);
    T_D1 = ones(1,size(VE_D1,2)).*size(VE_D1,1);
    T_D4 = ones(1,size(VE_D4,2)).*size(VE_D4,1);
    samp_freq = round(fs); % sampling frequency, Hz
    %% Downsample from 600 to 100Hz
    Hz = 100;
    [data_new_D1,T_new_D1] = downsampledata_(VE_D1_f',T_D1,Hz,samp_freq);
    [data_new_D4,T_new_D4] = downsampledata_(VE_D4_f',T_D4,Hz,samp_freq);

    load([path_hmm,file_hmm],'burst*', 'Gamma_D1_pad', 'Gamma_D4_pad', ...
        'num_bursts*', 'state_masks_D*', 'options', 'stim_win', 'post_win', ...
        'beta_envelope*', 'state_probabilities*', 'fit*')

    % get burst amplitude
    burst_starts = find(diff(burst_mask_D1) == 1);
    burst_ends = find(diff(burst_mask_D1) == -1);
    if burst_starts(1) > burst_ends(1)
        burst_ends(1) = [];
    end
    if burst_starts(end) > burst_ends(end)
        burst_starts(end) = [];
    end
    clear burst_max*
    for b = 1:length(burst_starts)
        burst_max_D1(b) = max(abs(data_new_D1(burst_starts(b):burst_ends(b))));
    end
    burst_amp_D1 = mean(burst_max_D1);
    burst_amp_D1_std = std(burst_max_D1);


    burst_starts = find(diff(burst_mask_D4) == 1);
    burst_ends = find(diff(burst_mask_D4) == -1);
    if burst_starts(1) > burst_ends(1)
        burst_ends(1) = [];
    end
    if burst_starts(end) > burst_ends(end)
        burst_starts(end) = [];
    end
    clear burst_max*
    for b = 1:length(burst_starts)
        burst_max_D4(b) = max(abs(data_new_D4(burst_starts(b):burst_ends(b))));
    end
    burst_amp_D4 = mean(burst_max_D4);
    burst_amp_D4_std = std(burst_max_D4);

    save([path_hmm,file_hmm_amp],"burst_amp_*")
else
    load([path_hmm,file_hmm_amp],"burst_amp_D1","burst_amp_D4","burst_amp_D1_std","burst_amp_D4_std");
end
end
%% Funcs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data_new,T_new] = downsampledata_(data,T,fs_new,fs_data)
%data should be n_T x channels
if fs_new == fs_data
    data_new = data; T_new = T; return
end

T_new = ceil((fs_new/fs_data) * T);
ndim = size(data,2);
data_new = zeros(sum(T_new),ndim);

for i=1:length(T)
    ind1 = sum(T(1:i-1))+ (1:T(i));
    ind2 = sum(T_new(1:i-1))+ (1:T_new(i));
    for n = 1:ndim
        data_new(ind2,n) = resample(data(ind1,n),fs_new,fs_data);
    end
end
end

function x = normalise_(x,dim)
% normalise(x)
% Removes the Average or mean value and makes the std=1
%
% normalise(X,DIM)
% Removes the mean and makes the std=1 along the dimension DIM of X.

if(nargin==1),
    dim = 1;
    if(size(x,1) > 1)
        dim = 1;
    elseif(size(x,2) > 1)
        dim = 2;
    end
end

dims = size(x);
dimsize = size(x,dim);
dimrep = ones(1,length(dims));
dimrep(dim) = dimsize;

x = x - repmat(mean(x,dim),dimrep);
x = x./repmat(std(x,0,dim),dimrep);
end

function plot_state_spectra_(fit,burst_state,sub)
plot(fit.state(1).f,fit.state(1).psd,'-','LineWidth',2);
hold on;
plot(fit.state(2).f,fit.state(2).psd,'-','LineWidth',2);
plot(fit.state(3).f,fit.state(3).psd,'-','LineWidth',2);
xlabel('Frequency (Hz)'); ylabel('P.S.D.');
title(['Sub ',sub,' | Burst state: ',num2str(burst_state)]);
set(gca,'FontSize',10); legend('state 1', 'state 2', 'state 3');
end