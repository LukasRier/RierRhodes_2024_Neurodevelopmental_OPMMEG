function peak_VE_hmm_func(sub,ses,project_dir)
overwrite = true
restoredefaultpath
% close all
clc
% project_dir =  'R:\DRS-KidsOPM\Paediatric_OPM_Notts_AdultData\';
%
% sub = '001';
% ses = '001';
run = 'run-001';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('.\HMM_bursts-master\')) %from https://github.com/ZSeedat/HMM_bursts/tree/master

addpath(genpath('.\HMM-MAR-master')) %from https://github.com/OHBA-analysis/HMM-MAR

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

if ~exist(path_hmm,'dir'); mkdir(path_hmm);end

%% Get 1-48 Hz filtered VE timecourse
S = load([path_Tstat,filesep,file_D1_VE],'VE_index');
VE_D1 = S.VE_index;
S = load([path_Tstat,filesep,file_D4_VE],'VE_pinky');
VE_D4 = S.VE_pinky;
clear S;


fs = 1200;
hp = 1;
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

if ~exist([path_hmm,file_hmm],'file') | overwrite
%%
% Set the HMM Options
lags = 11; % sensible to range between 3 and 11.
no_states = 3;
options = struct(); % Create options struct
options.K = no_states;
options.standardise = 1;
options.verbose = 1;
options.Fs = Hz; % the frequency of the downsampled data
options.order = 0;
options.embeddedlags = -lags:lags;
options.zeromean = 1;
options.covtype = 'full';
options.useMEX = 0; % runs much faster with the compiled mex files
options.dropstates = 1; % the HMM can drop states if there isn't
% sufficient evidence to justify this number of states.
options.DirichletDiag = 10; % diagonal of prior of trans-prob matrix (default 10 anyway)
options.useParallel = 0;
options.downsample = 0;
% HMM computation

data_reg_D1 = normalise_(data_new_D1,1); % normalise!!
[hmm_D1, Gamma_D1] = hmmmar(data_reg_D1,T_new_D1,options); % hmm inference

data_reg_D4 = normalise_(data_new_D4,1); % normalise!!
[hmm_D4, Gamma_D4] = hmmmar(data_reg_D4,T_new_D4,options); % hmm inference

% % Save the output
% % save([path_hmm,file_hmm], 'hmm_D*','Gamma_D*')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classify burst state
% Correct Gamma time-course for lags in AR model
Gamma_D1_pad = padGamma(Gamma_D1,T_new_D1,options);
Gamma_D4_pad = padGamma(Gamma_D4,T_new_D4,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Correlation of burst probability timecourse with beta envelope
% 13-30Hz data:
[wt_D1,wf] = cwt(data_reg_D1,'amor',Hz); % morlet wavelet
[wt_D4,~] = cwt(data_reg_D4,'amor',Hz); % morlet wavelet
beta_wavelet_freqs = wf > 13 & wf < 30;
% Envelope of beta oscillations
beta_envelope_D1 = mean(abs(wt_D1(beta_wavelet_freqs,:)),1)';
beta_envelope_D4 = mean(abs(wt_D4(beta_wavelet_freqs,:)),1)';
% 
% corr_tmp_D1 = [];
% for k = 1:size(Gamma_D1_pad,2)
%     corr_tmp_D1(k) = corr(Gamma_D1_pad(:,k),beta_envelope_D1);
% end
% [a, burst_state_D1] = max(corr_tmp_D1);
% 
% corr_tmp_D4 = [];
% for k = 1:size(Gamma_D4_pad,2)
%     corr_tmp_D4(k) = corr(Gamma_D4_pad(:,k),beta_envelope_D4);
% end
% [a, burst_state_D4] = max(corr_tmp_D4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classify bursts using thresholded gamma and stim vs post stim modulation
% of burst probability:

gamma_thresh = 2/3;
state_masks_D1 = Gamma_D1_pad>gamma_thresh;
state_masks_D4 = Gamma_D4_pad>gamma_thresh;

stim_win = round([0.3,0.8]*Hz);
post_win = round([1,1.5]*Hz);

state_probabilities_D1 = squeeze(mean(reshape(state_masks_D1,[],size(T_new_D1,2),3),2));
state_probabilities_D4 = squeeze(mean(reshape(state_masks_D4,[],size(T_new_D4,2),3),2));

probability_modulation_D1 = mean(state_probabilities_D1(stim_win(1):stim_win(2),:)) - ...
    mean(state_probabilities_D1(post_win(1):post_win(2),:));
probability_modulation_D4 = mean(state_probabilities_D4(stim_win(1):stim_win(2),:)) - ...
    mean(state_probabilities_D4(post_win(1):post_win(2),:));

[~,burst_state_D1] = min(probability_modulation_D1);
[~,burst_state_D4] = min(probability_modulation_D4);

burst_mask_D1 = Gamma_D1_pad(:,burst_state_D1)>gamma_thresh;
burst_mask_D4 = Gamma_D4_pad(:,burst_state_D4)>gamma_thresh;


%% Burst duration
% Burst lifetimes
LTs_hmmar = getStateLifeTimes(Gamma_D1_pad,size(Gamma_D1_pad,1),options,0,gamma_thresh);
burst_LTs_D1 = LTs_hmmar{burst_state_D1};
burst_dur_D1 = mean(burst_LTs_D1)*1000/options.Fs; % milliseconds

LTs_hmmar = getStateLifeTimes(Gamma_D4_pad,size(Gamma_D4_pad,1),options,0,gamma_thresh);
burst_LTs_D4 = LTs_hmmar{burst_state_D4};
burst_dur_D4 = mean(burst_LTs_D4)*1000/options.Fs; % milliseconds

%% Number of bursts per second
num_bursts_D1 = length(burst_LTs_D1)/(size(Gamma_D1_pad,1)/options.Fs);
num_bursts_D4 = length(burst_LTs_D4)/(size(Gamma_D4_pad,1)/options.Fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Time between bursts
state_ITs = getStateIntervalTimes(Gamma_D1_pad,size(Gamma_D1_pad,1),options,0,gamma_thresh);
burst_ITs_D1 = state_ITs{burst_state_D1};
burst_ISL_D1 = mean(burst_ITs_D1)*1000/options.Fs; % milliseconds

state_ITs = getStateIntervalTimes(Gamma_D4_pad,size(Gamma_D4_pad,1),options,0,gamma_thresh);
burst_ITs_D4 = state_ITs{burst_state_D4};
burst_ISL_D4 = mean(burst_ITs_D4)*1000/options.Fs; % milliseconds

%% State spectra
T = length(data_reg_D1);
fit_D1 = hmmspectramt(data_reg_D1,T,Gamma_D1_pad,options); % statewise multitaper
burst_state_spectra_D1= fit_D1.state(burst_state_D1).psd';
spec_f_D1 = fit_D1.state(burst_state_D1).f';
% plot_state_spectra_(fit_D1,burst_state_D1,sub)
% title(['Sub ',sub,' | D1 Burst state: ',num2str(burst_state_D1)]);

T = length(data_reg_D4);
fit_D4 = hmmspectramt(data_reg_D4,T,Gamma_D4_pad,options); % statewise multitaper
burst_state_spectra_D4= fit_D4.state(burst_state_D4).psd';
spec_f_D4 = fit_D4.state(burst_state_D4).f';
% plot_state_spectra_(fit_D4,burst_state_D4,sub)
% title(['Sub ',sub,' | D4 Burst state: ',num2str(burst_state_D4)]);

% combined non-burst spectra
T = length(data_reg_D1);

Gamma_D1_pad_binary = [Gamma_D1_pad(:,burst_state_D1),1-Gamma_D1_pad(:,burst_state_D1)]
fit_D1_binary = hmmspectramt(data_reg_D1,T,Gamma_D1_pad_binary,options); % statewise multitaper
burst_state_spectra_D1= fit_D1_binary.state(1).psd';
spec_f_D1 = fit_D1_binary.state(1).f';
% plot_state_spectra_(fit_D1,burst_state_D1,sub)
% title(['Sub ',sub,' | D1 Burst state: ',num2str(burst_state_D1)]);

T = length(data_reg_D4);
Gamma_D4_pad_binary = [Gamma_D4_pad(:,burst_state_D4),1-Gamma_D4_pad(:,burst_state_D4)]
fit_D4_binary = hmmspectramt(data_reg_D4,T,Gamma_D4_pad_binary,options); % statewise multitaper
burst_state_spectra_D4= fit_D4_binary.state(1).psd';
spec_f_D4 = fit_D4_binary.state(1).f';

save([path_hmm,file_hmm],'burst*', 'Gamma_D1_pad', 'Gamma_D4_pad', ...
    'num_bursts*', 'state_masks_D*', 'options', 'stim_win', 'post_win', ...
    'beta_envelope*', 'state_probabilities*', 'fit*')
%%
else
    load([path_hmm,file_hmm])
end
figure
set(gcf,'Color','w','Position', [680 71 842 907])
tiledlayout(5,6)
nexttile([1,3])
time = linspace(0,T_new_D1(1)./Hz,T_new_D1(1));
plot(time,mean(reshape(beta_envelope_D1,T_new_D1(1),[]),2),'k','LineWidth',3)
title('D1 cwt beta amplitude')

nexttile([1,3])
plot(time,mean(reshape(beta_envelope_D4,T_new_D4(1),[]),2),'k','LineWidth',3)
title('D4 cwt beta amplitude')

state_probabilities_D1 = squeeze(mean(reshape(state_masks_D1,[],size(T_new_D1,2),3),2));
state_probabilities_D4 = squeeze(mean(reshape(state_masks_D4,[],size(T_new_D4,2),3),2));
nexttile([1,3])
plot(time,state_probabilities_D1);title('D1 state probabilities')
nexttile([1,3])
plot(time,state_probabilities_D4);lh=legend({'State 1','State 2','State 3'});title('D4 state probabilities');lh.Position

raster = reshape(state_masks_D1,[],size(T_new_D1,2),3);
t=[1,1];
nexttile(t)
imagesc(raster(:,:,1)');title('D1 State1')
nexttile(t)
imagesc(raster(:,:,2)');title('D1 State2')
nexttile(t)
imagesc(raster(:,:,3)');title('D1 State3')

raster = reshape(state_masks_D4,[],size(T_new_D4,2),3);
nexttile(t)
imagesc(raster(:,:,1)');title('D4 State1')
nexttile(t)
imagesc(raster(:,:,2)');title('D4 State2')
nexttile(t)
imagesc(raster(:,:,3)');title('D4 State3')

nexttile([2,3])
plot_state_spectra_(fit_D1,burst_state_D1,sub)
nexttile([2,3])
plot_state_spectra_(fit_D4,burst_state_D4,sub)

end%func end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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