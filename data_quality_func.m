function data_quality = data_quality_func(sub,ses,project_dir)
restoredefaultpath
close all
clc
% project_dir = 'P:\Paediatric_OPM_Notts\';
% sub = '001';
% ses = '001';
run = 'run-001';
data_quality.ID = sub;
data_quality.task = 'braille';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fieldtrip_path = ['R:\DRS-KidsOPM\Paediatric_OPM_Notts\fieldtrip-20220906'];
if ~exist(fieldtrip_path,"dir")
    error("change 'fieldtrip_path' to valid path")
else
    addpath(fieldtrip_path)
end
script_dir = mfilename('fullpath');fname = mfilename;script_dir = script_dir(1:end-length(fname));
addpath(script_dir)
addpath([script_dir,'Beamformer',filesep,''])
ft_defaults;
datadir = [project_dir,'Data',filesep,'BIDS',filesep];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_type = '_task-braille';
filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];
path_main = [datadir,'sub-',sub,filesep,'ses-',ses,filesep];

path_ICA = [datadir,'derivatives',filesep,'ICA',filesep,'sub-',sub,filesep];
if ~exist(path_ICA,'dir'); mkdir(path_ICA);end
files_ICA = [path_ICA,filename];

path_cleaning = [datadir,'derivatives',filesep,'cleaning',filesep,'sub-',sub,filesep];
if ~exist(path_cleaning,'dir'); mkdir(path_cleaning);end
files_cleaning = [path_cleaning,filename];

path_VEs = [datadir,'derivatives',filesep,'VEs',filesep,'sub-',sub,filesep];
if ~exist(path_VEs,'dir'); mkdir(path_VEs);end

path_meg_data = [path_main,'meg',filesep];

path_meshes = [datadir,'derivatives',filesep,'sourcespace',filesep,'sub-',sub,filesep];
files_meshes = ['sub-',sub,'_meshes.mat'];
files_AAL_centroids = ['sub-',sub,'_AAL_centroids.nii.gz'];
files_AAL_regions = ['sub-',sub,'_AAL_regions.nii.gz'];
files_VOI = ['sub-',sub,'_AAL_VOI.mat'];

path_mri = [path_main,'anat',filesep];
files_mri = ['sub-',sub,'_anat.nii'];
S.mri_file = [path_mri,files_mri];

path_helmet = [datadir,'derivatives',filesep,'helmet',filesep,'sub-',sub,filesep];
files_helmet_info = dir([path_helmet,'*.mat']);files_helmet_info=files_helmet_info.name;

files_voxlox = ['sub-',sub,'_AAL_locs.mat'];
files_channels = [filename,'_channels.tsv'];

files_events = [filename,'_events.tsv'];

path_Tstat = [datadir,'derivatives',filesep,'Tstats',filesep,'sub-',sub,filesep]
if ~exist(path_Tstat,'dir'); mkdir(path_Tstat);end
files_Tstat = [filename,'_pseudoT_'];


%read meg data
cd(path_meg_data)
read_info = readlines([filename,'_meg_read_info.txt']);
Size = strsplit(read_info(1));Size = [str2num(Size(2)),str2num(Size(4))];

Precision = strsplit(read_info(2));Precision = Precision(2);

Ordering = strsplit(read_info(3));Ordering = Ordering(2);

FileID = fopen([path_meg_data,filename,'_meg.dat'],'r');

data=fread(FileID,Size,lower(Precision),Ordering)';
fclose(FileID);

% fs and other info
fID = fopen([path_meg_data,filename,'_meg.json']);
raw = fread(fID,inf);
json_info = jsondecode(char(raw'));
fs = json_info.SamplingFrequency;

% trigger info
event_table = readtable([path_meg_data,filename,'_events.tsv'],'FileType','text','delimiter','\t');
all_start_samps = [round(event_table(startsWith(event_table.type(:),'Start_index'),:).sample);...
    round(event_table(startsWith(event_table.type(:),'Start_pinky'),:).sample)];

%helmet info
load([path_helmet,files_helmet_info])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Preproc
% Mean correct
data = data - mean(data,1);

% Notch filter
for harms = [50,100,150]
    Wo = harms/(fs/2);  BW = Wo/35;
    [b,a] = iirnotch(Wo,BW);
    disp('Applying Notch filter')
    data = filter(b,a,data,[],1);
end
Nchans = size(data,2);

% bandpass filter for viewing
disp('Applying 1-150 Hz bandpass filter')

hp = 1;
lp = 150;
[b,a] = butter(4,2*[hp lp]/fs);
data_f = [filtfilt(b,a,data)]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% bad channels

ch_table = readtable([path_meg_data,files_channels(1:end-4),'_proc.tsv'],...
    'Delimiter','tab','FileType','text');

%% Somewhat hacky way of getting slot numbers for later. Change ICA comp visualiser?

precision_limit = 1e-6;
 
pos = [ch_table.Px,ch_table.Py,ch_table.Pz];
for sl_i = 1:size(Helmet_info.lay.pos,1)
    detected_ind = find(sqrt(sum((repmat(Helmet_info.sens_pos(sl_i,:),height(ch_table),1) - pos/100).^2,2)) < precision_limit)
    if ~isempty(detected_ind)
        ch_table.slot_no(detected_ind) = sl_i;
    end
end
%%
% remove from data matrix
disp("Removing bad channels")
bad_chans_data = find(startsWith(ch_table.status,'bad'));

ch_table(bad_chans_data,:) = [];
data_f(bad_chans_data,:) = [];
data_quality.N_bad_chans = length(bad_chans_data)
%% sensor info
S.sensor_info.pos = [ch_table.Px,ch_table.Py,ch_table.Pz]./100;
S.sensor_info.ors = [ch_table.Ox,ch_table.Oy,ch_table.Oz];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

xfm = load(sprintf('sub-%s_ses-%s%s_%s_sens2template_transform.txt',sub,ses,exp_type,run));
xfm = reshape(xfm,4,4);
xfm_rot = xfm(1:3,1:3);
xfm_translation = xfm(1:3,4)';
S.sensor_info.pos = (xfm_rot*S.sensor_info.pos' + xfm_translation')';
S.sensor_info.ors = (xfm_rot*S.sensor_info.ors')';
%% Epoch data
% segment using trigger
disp("Chopping data into epochs")
epoch_length = 3.5;
events_table = readtable([path_meg_data,files_events],'FileType','text','Delimiter','tab');
index_events = startsWith(events_table.type,'Start_index');
pinky_events = startsWith(events_table.type,'Start_pinky');
index_and_pinky_events = find(index_events|pinky_events);
start_samples = events_table.sample(index_and_pinky_events);
end_samples = start_samples + epoch_length*fs-1;
%check whether last trial is fully available and discard if not
unfinished_trls = find(end_samples > size(data_f,2));
start_samples(unfinished_trls) = [];
end_samples(unfinished_trls) = [];
index_and_pinky_events(unfinished_trls) = [];
data_f_mat = zeros(size(data_f,1),epoch_length*fs,size(end_samples,1));
for tr_i = 1:size(end_samples,1)
    data_f_mat(:,:,tr_i) = data_f(:,start_samples(tr_i):end_samples(tr_i));
end

clear data_f
%%%%%

%% Put data in FT format
disp("Converting to FieldTrip format")
[data_strct] = makeFTstruct(data_f_mat,fs,ch_table,S.sensor_info);
%% trial info
data_strct = ft_checkdata(data_strct, 'datatype', 'raw', 'hassampleinfo', 'yes');

trl_mat = sampleinfo2trl(data_strct);
trial_info = table(trl_mat(:,1),trl_mat(:,2),trl_mat(:,3),'VariableNames',{'start','end','offset'})
trial_info.type = events_table.type(index_and_pinky_events)
removefields(data_strct,'sampleinfo')

%% artefacts
load([files_cleaning,'_vis_artfcts.mat'],'vis_artfcts')


% automatic artifact rejection
thresh_val = 3;
auto_artfcts = get_bad_segments(data_f_mat,thresh_val);
fprintf('Found %d artifacts using a threshold of %d std. deviations.\n',...
    size(auto_artfcts,1),thresh_val);
data_quality.N_auto_bad_trl = size(auto_artfcts,1);
% combine artifacts and reject bad trials
cfg = [];
cfg.artfctdef.visual.artifact = [vis_artfcts;auto_artfcts];
cfg.artfctdef.reject  = 'complete';
data_vis_clean = ft_rejectartifact(cfg,data_strct);

fprintf('\nRejected %d of %d epochs of length %1.2f s.\n',...
    size(data_strct.trial,2)-size(data_vis_clean.trial,2),size(data_strct.trial,2),epoch_length);
data_quality.N_tot_bad_trl = size(data_strct.trial,2)-size(data_vis_clean.trial,2);
data_quality.N_tot_trl = size(data_strct.trial,2);
good_trials = false(height(trial_info),1);
for clean_trial_ind = 1:size(data_vis_clean.cfg.trl,1)
    good_trials(sum(data_vis_clean.cfg.trl(clean_trial_ind,1:2)==trial_info{:,1:2},2)==2) = true;
end

%% add 9 fake bad trials
good_trial_inds = find(good_trials);
final_9_good_trls = good_trial_inds(end-8:end)
vis_artfcts = [vis_artfcts;[trial_info.start(final_9_good_trls)+100,trial_info.end(final_9_good_trls)-100]]
save([files_cleaning,'_vis_artfcts_adj.mat'],'vis_artfcts')

% only include good trials in info table
trial_info = trial_info(good_trials,:);
data_quality.good_chans = size(data_vis_clean.trial{1},1);
data_quality.good_trls_D2 = sum(endsWith(trial_info.type,'index'));
data_quality.good_trls_D5 = sum(endsWith(trial_info.type,'pinky'));

%% add 9 fake bad trials
good_trial_inds = find(good_trials);
final_9_good_trls = good_trial_inds(end-8:end)

%% ICA
% lay = Helmet_info.lay;
% disp("ICA artifact rejection")
% 
% disp("Loading bad coponents, topographies and old unmixing matrix")
% load([files_ICA,'_bad_ICA_comps.mat'],'bad_comps')
% data_quality.bad_ICA_comps = length(bad_comps)  

data_quality = struct2table(data_quality)

end

function trl = sampleinfo2trl(data)
% borrowed from private FieldTrip funcs
% SAMPLEINFO2TRL constructs the trial definition from the sampleinfo, the time axes
% and optionally from the trialinfo
%
% Use as
%   trl = sampleinfo2trl(data)
%

% get the begin and end sample of each trial
begsample = data.sampleinfo(:,1);
endsample = data.sampleinfo(:,2);

% recreate the offset
offset = zeros(numel(data.trial), 1);
for i=1:numel(data.trial)
    offset(i) = round(data.time{i}(1)*data.fsample);
end

if isfield(data, 'trialinfo') && istable(data.trialinfo)
    trl = table(begsample, endsample, offset);
    trl = horzcat(trl, data.trialinfo);
elseif isfield(data, 'trialinfo') && isnumeric(data.trialinfo)
    trl = [begsample endsample offset data.trialinfo];
else
    trl = [begsample endsample offset];
end
end

