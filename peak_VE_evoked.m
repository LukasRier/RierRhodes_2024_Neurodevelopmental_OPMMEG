function peak_VE_evoked(sub,ses,project_dir)
restoredefaultpath
% close all
clc
% project_dir =  'R:\DRS-KidsOPM\Paediatric_OPM_Notts_AdultData\';
%
% sub = '001';
% ses = '001';
run = 'run-001';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
path_evoked = [datadir,'derivatives',filesep,'evoked',filesep,'sub-',sub,filesep]
if ~exist(path_evoked,'dir'); mkdir(path_evoked);end
files_peak_evoked = [filename,'_peak_evoked.mat'];

%% Get 1-48 Hz filtered VE timecourse
S = load([path_Tstat,filesep,file_D1_VE],'VE_index');
VE_D1 = S.VE_index;
S = load([path_Tstat,filesep,file_D4_VE],'VE_pinky');
VE_D4 = S.VE_pinky;
clear S;


fs = 1200;
hp = 4;
lp = 40;
[b,a] = butter(4,2*[hp lp]/fs);
VE_D1_f = [filtfilt(b,a,VE_D1)];
VE_D4_f = [filtfilt(b,a,VE_D4)];

index_evoked = mean(VE_D1_f,2);
little_evoked = mean(VE_D4_f,2);

figure(1)
time = linspace(0,length(little_evoked)/fs,length(little_evoked));
subplot(2,1,1)
plot(time,abs(index_evoked),'k','LineWidth',0.001)
hold on
yline(0,'k')
xline(0.132,'r')

subplot(2,1,2)
plot(time,abs(little_evoked),'k','LineWidth',0.001)
hold on
yline(0,'k')
xline(0.132,'r')

figure(2)
time = linspace(0,length(little_evoked)/fs,length(little_evoked));
subplot(2,1,1)
plot(time,(index_evoked),'k','LineWidth',0.001)
hold on
yline(0,'k')
xline(0.132,'r')

subplot(2,1,2)
plot(time,(little_evoked),'k','LineWidth',0.001)
hold on
yline(0,'k')
xline(0.132,'r')
save([path_evoked,files_peak_evoked],'index_evoked','little_evoked')

end
%% Funcs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
