function index_pinky_4mm_Tstat_func(sub,ses,project_dir)
restoredefaultpath
cleaning_only = 0;
fake_bad_ch =1;

close all
clc
% project_dir = 'P:\Paediatric_OPM_Notts\';
% sub = '001';
% ses = '001';
run = 'run-001';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fieldtrip_path = [project_dir,filesep,'fieldtrip-20220906'];
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
files_brain = ['sub-',sub,'_brain.nii'];
files_VOI = ['sub-',sub,'_AAL_VOI.mat'];

path_mri = [path_main,'anat',filesep];
files_mri = ['sub-',sub,'_anat.nii'];
S.mri_file = [path_mri,files_mri];

path_helmet = [datadir,'derivatives',filesep,'helmet',filesep,'sub-',sub,filesep];
files_helmet_info = dir([path_helmet,'*.mat']);files_helmet_info=files_helmet_info.name;

files_voxlox = ['sub-',sub,'_AAL_locs.mat'];
files_channels = [filename,'_channels.tsv'];

files_events = [filename,'_events.tsv'];

path_Tstat = [datadir,'derivatives',filesep,'Tstats_4mm',filesep,'sub-',sub,filesep]
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

% Get rid of bad channels
%% Get rid of bad channels

if ~exist([path_meg_data,files_channels(1:end-4),'_proc.tsv'],'file')
    %     error("Use 'get_good_channels.m for this dataset first")
    ch_table = readtable([path_meg_data,files_channels],'FileType','text','Delimiter','tab');
    ch_table.isx = endsWith(ch_table.name,'X');
    ch_table.isy = endsWith(ch_table.name,'Y');
    ch_table.isz = endsWith(ch_table.name,'Z');
    ch_table.slot_no = zeros(height(ch_table),1);
    % sanity check
    if sum(sum([ch_table.isx,ch_table.isy,ch_table.isz],2)) ~= height(ch_table)
        error('Channel orientation [x,y,z] labels might be wrong!')
    end
    [ch_table] = Bad_Channels(data_f',ch_table,fs);
    writetable(ch_table,[path_meg_data,files_channels(1:end-4),'_proc.tsv'],...
        'WriteRowNames',true,'Delimiter','tab','FileType','text')
else
    ch_table = readtable([path_meg_data,files_channels(1:end-4),'_proc.tsv'],...
        'Delimiter','tab','FileType','text');
end

%% Somewhat hacky way of getting slot numbers for later. Change ICA comp visualiser?

% ors = [ch_table.Ox,ch_table.Oy,ch_table.Oz];
precision_limit = 1e-6;
% for sl_i = 1:size(Helmet_info.lay.pos,1)
%     detected_ind_X = find(sum(abs(repmat(Helmet_info.sens_ors_X(sl_i,:),height(ch_table),1) - ors),2)<precision_limit)
%     detected_ind_Y = find(sum(abs(repmat(Helmet_info.sens_ors_Y(sl_i,:),height(ch_table),1) - ors),2)<precision_limit)
%     detected_ind_Z = find(sum(abs(repmat(Helmet_info.sens_ors_Z(sl_i,:),height(ch_table),1) - ors),2)<precision_limit)
%     if ~isempty(detected_ind_X)
%         if(ch_table.slot_no(detected_ind_X) == 0)
%             ch_table.slot_no(detected_ind_X) = sl_i;
%         else
%             error('Duplicate slot detected!')
%         end
%     end
%     
%     if ~isempty(detected_ind_Y)
%         if(ch_table.slot_no(detected_ind_Y) == 0)
%             ch_table.slot_no(detected_ind_Y) = sl_i;
%         else
%             error('Duplicate slot detected!')
%         end
%     end
%     
%     if ~isempty(detected_ind_Z)
%         if(ch_table.slot_no(detected_ind_Z) == 0)
%             ch_table.slot_no(detected_ind_Z) = sl_i;
%         else
%             error('Duplicate slot detected!')
%         end
%     end
% end
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
%% Mean field correction matrix
N = S.sensor_info.ors; % orientation matrix (N_sens x 3)
S.M = eye(length(N)) - N*pinv(N);

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
%% resample for viewing and mark artefacts
disp("Check for bad trials")
if ~exist([files_cleaning,'_vis_artfcts.mat'],'file')
    resamplefs = 150; %Hz
    cfg            = [];
    cfg.resamplefs = resamplefs;
    cfg.detrend    = 'no';
    data_preproc_150   = ft_resampledata(cfg, data_strct);
    
    %%% Get rid of bad trials
    cfg_art          = [];
    cfg_art.viewmode = 'vertical';
    cfg_art = ft_databrowser(cfg_art,data_preproc_150);
    clear data_preproc_150
    
    vis_artfcts = cfg_art.artfctdef.visual.artifact * ...
        (data_strct.fsample/resamplefs);
    save([files_cleaning,'_vis_artfcts.mat'],'vis_artfcts')
else
    if exist([files_cleaning,'_vis_artfcts_adj.mat']) && fake_bad_ch
        load([files_cleaning,'_vis_artfcts_adj.mat'],'vis_artfcts')
    else
        load([files_cleaning,'_vis_artfcts.mat'],'vis_artfcts')
    end
end

% automatic artifact rejection
thresh_val = 3;
auto_artfcts = get_bad_segments(data_f_mat,thresh_val);
fprintf('Found %d artifacts using a threshold of %d std. deviations.\n',...
    size(auto_artfcts,1),thresh_val);

% combine artifacts and reject bad trials
cfg = [];
cfg.artfctdef.visual.artifact = [vis_artfcts;auto_artfcts];
cfg.artfctdef.reject  = 'complete';
data_vis_clean = ft_rejectartifact(cfg,data_strct);

fprintf('\nRejected %d of %d epochs of length %1.2f s.\n',...
    size(data_strct.trial,2)-size(data_vis_clean.trial,2),size(data_strct.trial,2),epoch_length);

good_trials = false(height(trial_info),1);
for clean_trial_ind = 1:size(data_vis_clean.cfg.trl,1)
    good_trials(sum(data_vis_clean.cfg.trl(clean_trial_ind,1:2)==trial_info{:,1:2},2)==2) = true;
end
% only include good trials in info table
trial_info = trial_info(good_trials,:);
%% ICA
lay = Helmet_info.lay;
disp("ICA artifact rejection")
if ~exist([files_ICA,'_bad_ICA_comps.mat'],'file') || ~exist([files_ICA,'_ICA_data.mat'],'file')
    
    % Resample for faster ICA
    cfg            = [];
    cfg.resamplefs = 150;
    cfg.detrend    = 'no';
    data_ica_150   = ft_resampledata(cfg, data_vis_clean);
    
    % Run ICA on 150 Hz data or load previous unmixing matrix
    cfg            = [];
    if ~exist([files_ICA,'_ICA_data.mat'],'file')
        cfg.method = 'runica';
    else
        load([files_ICA,'_ICA_data.mat'],'comp150')
        cfg.unmixing   = comp150.unmixing;
        cfg.topolabel  = comp150.topolabel;
    end
    comp150    = ft_componentanalysis(cfg, data_ica_150);
    
    % Inspect components for rejection or load file with bad component list
    if ~exist([files_ICA,'_bad_ICA_comps.mat'],'file')
        disp("Choose bad components")
        [bad_comps] = plot_ICA_comps(comp150,ch_table,lay,[]);
        
        save([files_ICA,'_bad_ICA_comps.mat'],'bad_comps')
        close(gcf)
    else
        disp("Loading saved bad components")
        load([files_ICA,'_bad_ICA_comps.mat'],'bad_comps')
    end
    
    % only keep unmixing matrix and topolabel for component removal
    tokeep = {'unmixing','topolabel'};
    fns=fieldnames(comp150);
    toRemove = fns(~ismember(fns,tokeep));
    comp150 = rmfield(comp150,toRemove);
    save([files_ICA,'_ICA_data.mat'],'comp150')
    
    clear data_ica_150
else
    disp("Loading bad coponents, topographies and old unmixing matrix")
    load([files_ICA,'_bad_ICA_comps.mat'],'bad_comps')
    load([files_ICA,'_ICA_data.mat'],'comp150')
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Perform ICA on original 1200 Hz data using unmixing matrix and topolabel
cfg            = [];
cfg.unmixing   = comp150.unmixing;
cfg.topolabel  = comp150.topolabel;
comp1200        = ft_componentanalysis(cfg, data_vis_clean);

% Plot comps again to confirm they are correct
disp("Confirm bad ICA components")
% [bad_comps] = plot_ICA_comps(comp1200,ch_table,lay,bad_comps)
save([files_ICA,'_bad_ICA_comps.mat'],'bad_comps');

% Remove components from data
cfg           = [];
cfg.component = bad_comps;
data_ica_clean    = ft_rejectcomponent(cfg, comp1200,data_vis_clean);
N_clean_trls = size(data_ica_clean.trial,2);

% Plot  example sensor time courses pre and post ica

% figure(111);clf
% plot(data_vis_clean.trial{1,1}(contains(data_vis_clean.label,'LR [Z]'),:),'Color',[0.5,0.5,0.5])
% hold on
% plot(data_ica_clean.trial{1,1}(contains(data_vis_clean.label,'LR [Z]'),:),'Color',[0.9,0.5,0.5])
% xlabel('t/s')
% legend('Pre ICA','Post ICA')
clear data_vis_clean
%% Reconstitute data from FT structure

data_f_clean = [data_ica_clean.trial{1,:}];
clear data_ica_clean
% apply mean field correction
disp("Applying mean field correction")
data_f_clean = S.M*data_f_clean;

%% Further steps:
%
% Filter to band of interest
% Get source model info and beamformer location etc
% Beamform... spit out VEs

%% Source positions

% load AAL locations
VOI_mat_file = [path_meshes,files_VOI];
brain_1mm = ft_read_mri([path_meshes,files_brain]);
brain_1mm = ft_convert_units(brain_1mm,'m');
cfg = [];
cfg.downsample = 4;
[downsample_brain] = ft_volumedownsample(cfg, brain_1mm);
ft_write_mri([path_meshes,files_brain(1:end-4),'_4mm.nii'],downsample_brain,...
    'dataformat','nifti');
VOI = downsample_brain.anatomy > 0;

[sourcepos_vox(:,1),sourcepos_vox(:,2),sourcepos_vox(:,3)] = ind2sub(downsample_brain.dim, find(VOI));

sourcepos = ft_warp_apply(downsample_brain.transform,sourcepos_vox);


if ~cleaning_only
    
    [bf_outs_shell] = run_beamformer('shell',sourcepos,S,0,[],1);
    lead_fields_shell_xyz = bf_outs_shell.LF;
    
    outside_locs = squeeze(sum(lead_fields_shell_xyz(1,:,:)==0)==3);
    lead_fields_shell_xyz(:,:,outside_locs) = [];
    sourcepos(outside_locs,:)=[];
    sourcepos_vox(outside_locs,:)=[];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% convert orientation of sources to polar
    % Load meshes
    
    load([path_meshes,files_meshes],'meshes');
    
    meshes = ft_convert_units(meshes,'m');
    
    X = meshes.pnt; Origin = mean(X,1);
    Ndips = length(sourcepos);
    for n = 1:Ndips
        thispos = sourcepos(n,:);
        [phi,theta1,~] = cart2sph(thispos(1) - Origin(1),thispos(2) - Origin(2) ,thispos(3) - Origin(3));
        theta = pi/2 - theta1;
        Src_Or_theta(n,:) = [cos(theta)*cos(phi) cos(theta)*sin(phi) -sin(theta)];
        Src_Or_phi(n,:) = [-sin(phi) cos(phi) 0];
        Lead_fields(:,1,n) = Src_Or_theta(n,1)*lead_fields_shell_xyz(:,1,n) + Src_Or_theta(n,2)*lead_fields_shell_xyz(:,2,n) + Src_Or_theta(n,3)*lead_fields_shell_xyz(:,3,n);
        Lead_fields(:,2,n) = Src_Or_phi(n,1)*lead_fields_shell_xyz(:,1,n) + Src_Or_phi(n,2)*lead_fields_shell_xyz(:,2,n) + Src_Or_phi(n,3)*lead_fields_shell_xyz(:,3,n);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% make a plot of the geometry...
    figure(1);
    ft_plot_mesh(meshes,'facecolor',[.5 .5 .5],'facealpha',.3,'edgecolor','none')
    hold on
    scatter3(sourcepos(:,1),sourcepos(:,2),sourcepos(:,3),'ro','linewidth',3)
    view([130,30])
    fig = gcf;
    fig.Color = [1,1,1];
    plot3(S.sensor_info.pos(:,1),S.sensor_info.pos(:,2),S.sensor_info.pos(:,3),'o')
    quiver3(S.sensor_info.pos(ch_table.isx==1,1),S.sensor_info.pos(ch_table.isx==1,2),S.sensor_info.pos(ch_table.isx==1,3),...
        S.sensor_info.ors(ch_table.isx==1,1),S.sensor_info.ors(ch_table.isx==1,2),S.sensor_info.ors(ch_table.isx==1,3),'r','linewidth',2)
    quiver3(S.sensor_info.pos(ch_table.isy==1,1),S.sensor_info.pos(ch_table.isy==1,2),S.sensor_info.pos(ch_table.isy==1,3),...
        S.sensor_info.ors(ch_table.isy==1,1),S.sensor_info.ors(ch_table.isy==1,2),S.sensor_info.ors(ch_table.isy==1,3),'g','linewidth',2)
    quiver3(S.sensor_info.pos(ch_table.isz==1,1),S.sensor_info.pos(ch_table.isz==1,2),S.sensor_info.pos(ch_table.isz==1,3),...
        S.sensor_info.ors(ch_table.isz==1,1),S.sensor_info.ors(ch_table.isz==1,2),S.sensor_info.ors(ch_table.isz==1,3),'b','linewidth',2)
    plot3(Origin(1),Origin(2),Origin(3),'bo','linewidth',4)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% take a random lead field and plot it...
    figure(2);
    ft_plot_mesh(meshes,'facecolor',[.5 .5 .5],'facealpha',.3,'edgecolor','none')
    hold on
    ft_plot_topo3d(double(S.sensor_info.pos(ch_table.isz==1,:)),Lead_fields(ch_table.isz==1,2,16))
    alpha(gca,0.5)
    plot3(S.sensor_info.pos(ch_table.isx==1,1),S.sensor_info.pos(ch_table.isx==1,2),S.sensor_info.pos(ch_table.isx==1,3),'go','linewidth',3)
    
    [k2,~] = convhull(sourcepos,'Simplify',true);
    trisurf(k2,sourcepos(:,1),sourcepos(:,2),sourcepos(:,3),'FaceColor','m','FaceAlpha',0.5,'EdgeColor','none')
    scatter3(sourcepos(16,1),sourcepos(16,2),sourcepos(16,3),'r','linewidth',4)
    % quiver3(S.sensor_info.pos(ch_table.isx==1,1),S.sensor_info.pos(ch_table.isx==1,2),S.sensor_info.pos(ch_table.isx==1,3),...
    %     S.sensor_info.ors(ch_table.isx==1,1).*Lead_fields(ch_table.isx==1,2,16) + S.sensor_info.ors(ch_table.isy==1,1).*Lead_fields(ch_table.isy==1,2,16) + S.sensor_info.ors(ch_table.isz==1,1).*Lead_fields(ch_table.isz==1,2,16),...
    %     S.sensor_info.ors(ch_table.isx==1,2).*Lead_fields(ch_table.isx==1,2,16) + S.sensor_info.ors(ch_table.isy==1,2).*Lead_fields(ch_table.isy==1,2,16) + S.sensor_info.ors(ch_table.isz==1,2).*Lead_fields(ch_table.isz==1,2,16),...
    %     S.sensor_info.ors(ch_table.isx==1,3).*Lead_fields(ch_table.isx==1,2,16) + S.sensor_info.ors(ch_table.isy==1,3).*Lead_fields(ch_table.isy==1,2,16) + S.sensor_info.ors(ch_table.isz==1,3).*Lead_fields(ch_table.isz==1,2,16),'r','linewidth',2)
    quiver3(S.sensor_info.pos(ch_table.isz==1,1),S.sensor_info.pos(ch_table.isz==1,2),S.sensor_info.pos(ch_table.isz==1,3),...
        S.sensor_info.ors(ch_table.isz==1,1).*Lead_fields(ch_table.isz==1,2,16),...
        S.sensor_info.ors(ch_table.isz==1,2).*Lead_fields(ch_table.isz==1,2,16),...
        S.sensor_info.ors(ch_table.isz==1,3).*Lead_fields(ch_table.isz==1,2,16),'r','linewidth',2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clearvars -except path_* files_* downsample_brain sub ses epoch_length data_f_clean cleaning_only fs sourcepos sourcepos_vox Lead_fields N_clean_trls trial_info meshes
    %%
    %% filter the OPM data to band of interest
%     hp = 8;
%     lp = 25;   
    hp = 13;
    lp = 30;
    %         if ~exist(sprintf('%s%s_%d_%d_Hz_Z.mat',path_AEC,files_AEC,hp,lp),'file')
    [b,a] = butter(4,2*[hp lp]/fs);
    data_f = [filtfilt(b,a,data_f_clean')]';
    data_f_mat = reshape(data_f,size(data_f,1),[],height(trial_info));
    
    
    active_window = round([0.3,0.8].*fs);active_inds = active_window(1):active_window(2);
    control_window = round([2.5,3].*fs);control_inds = control_window(1):control_window(2);
    
    %index
    Digit1_trials = data_f_mat(:,:,startsWith(trial_info.type,'Start_index'));
    N_Digit1_trials = size(Digit1_trials,3);
    
    C_data_f_Digit1 = cov(reshape(Digit1_trials,size(Digit1_trials,1),prod(size(Digit1_trials,2,3)))');
    
    Ca_Digit1 = cov(reshape(Digit1_trials(:,active_inds,:),...
        size(Digit1_trials(:,active_inds,:),1),prod(size(Digit1_trials(:,active_inds,:),2,3)))');
    
    Cc_Digit1 = cov(reshape(Digit1_trials(:,control_inds,:),...
        size(Digit1_trials(:,control_inds,:),1),prod(size(Digit1_trials(:,control_inds,:),2,3)))');
    
    %pinky
    Digit4_trials = data_f_mat(:,:,startsWith(trial_info.type,'Start_pinky'));
    N_Digit4_trials = size(Digit4_trials,3);
    
    
    C_data_f_Digit4 = cov(reshape(Digit4_trials,size(Digit4_trials,1),prod(size(Digit4_trials,2,3)))');
    
    Ca_Digit4 = cov(reshape(Digit4_trials(:,active_inds,:),...
        size(Digit4_trials(:,active_inds,:),1),prod(size(Digit4_trials(:,active_inds,:),2,3)))');
    
    Cc_Digit4 = cov(reshape(Digit4_trials(:,control_inds,:),...
        size(Digit4_trials(:,control_inds,:),1),prod(size(Digit4_trials(:,control_inds,:),2,3)))');
    
    %TODO
    % - separate trials u/
    % - get Cov for each finger, invert and find weights as below
    % - get active and rest covariances u/
    % - estimate pseudo T like Barrat: T_D1 = w^TCaw - w^TCcw / 2w^Tw
    % - maybe runbeamformer with Tstat option on?
    % - find minimum activation (desync) for each finger and generate
    %   broadband VE for those locations to show desync in TFS or
    %   narrowband to show simple desync timecourse
    % - plot brain with both locs
    %% Beamform
    % Regularisation parameter
    mu = 0.05;
    % Index
    pseudoT_D1 = get_pseudoT(C_data_f_Digit1,Ca_Digit1,Cc_Digit1,Lead_fields,mu);
    %
    
    cax_val = [min(pseudoT_D1), 0.5.*min(pseudoT_D1)];
    [~, min_voxel_D1] = min(pseudoT_D1);
    fg = figure;
    fg.Name = 'D1'
    ft_plot_mesh(meshes,'facecolor',[.5 .5 .5],'facealpha',.1,'edgecolor','none')
    hold on
    scatter3(sourcepos(:,1),sourcepos(:,2),sourcepos(:,3),50,pseudoT_D1,'filled','MarkerFaceAlpha',0.2)
    colormap hot;caxis([cax_val]);cb = colorbar;cb.Label.String = 'Tstat';
    axis equal
    view([-120 20])
    fig = gcf;
    fig.Color = 'w';
    ax = gca;
    ax.FontSize = 14;
    
    % Find voxel with min/max value of Tstat
    dip_loc_D1 = sourcepos(min_voxel_D1,:);
    plot3(dip_loc_D1(1),dip_loc_D1(2),dip_loc_D1(3),'gx','markersize',10,'LineWidth',5)
    
    % get peak voxel timecourse
    %%
    [mean_Env_index] = get_envelope(Digit1_trials,C_data_f_Digit1,Lead_fields,min_voxel_D1,control_inds,mu);
    [mean_Env_index_pinkytrls] = get_envelope(Digit4_trials,C_data_f_Digit4,Lead_fields,min_voxel_D1,control_inds,mu);
    figure
    set(gcf,'Color','w');

    time = linspace(0,size(mean_Env_index,1)./fs,size(mean_Env_index,1));
    plot(time,mean_Env_index,'r','Displayname','D1 desync peak');
    hold on
    plot(time,mean_Env_index_pinkytrls,'r--','Displayname','D1 desync peak, D4 trials');
    legend
    
    %% Pinky
    pseudoT_D4 = get_pseudoT(C_data_f_Digit4,Ca_Digit4,Cc_Digit4,Lead_fields,mu);
    %
    
    cax_val = [min(pseudoT_D4), 0.5.*min(pseudoT_D4)];
    [~, min_voxel_D4] = min(pseudoT_D4);
    fg = figure;
    fg.Name = 'D4'
    ft_plot_mesh(meshes,'facecolor',[.5 .5 .5],'facealpha',.1,'edgecolor','none')
    hold on
    scatter3(sourcepos(:,1),sourcepos(:,2),sourcepos(:,3),50,pseudoT_D4,'filled','MarkerFaceAlpha',0.2)
    colormap hot;caxis([cax_val]);cb = colorbar;cb.Label.String = 'Tstat';
    axis equal
    view([-120 30])
    fig = gcf;
    fig.Color = 'w';
    ax = gca;
    ax.FontSize = 14;
    
    % Find voxel with min/max value of Tstat
    dip_loc_D4 = sourcepos(min_voxel_D4,:);
    plot3(dip_loc_D4(1),dip_loc_D4(2),dip_loc_D4(3),'gx','markersize',10,'LineWidth',5)
    
    % get peak voxel timecourse
    %%
    [mean_Env_pinky] = get_envelope(Digit4_trials,C_data_f_Digit4,Lead_fields,min_voxel_D4,control_inds,mu);
    [mean_Env_pinky_indextrls] = get_envelope(Digit1_trials,C_data_f_Digit1,Lead_fields,min_voxel_D4,control_inds,mu);
    figure
    set(gcf,'Color','w');
    plot(time,mean_Env_pinky,'b','Displayname','D4 desync peak');
    hold on
    plot(time,mean_Env_pinky_indextrls,'b--','Displayname','D4 desync peak, D1 trials');
    legend
    %%
    % save timecourses
    save([path_Tstat,'D1_peak_env_',num2str(hp),'_',num2str(lp),'_Hz.mat'],'mean_Env_index','mean_Env_index_pinkytrls')
    save([path_Tstat,'D4_peak_env_',num2str(hp),'_',num2str(lp),'_Hz.mat'],'mean_Env_pinky','mean_Env_pinky_indextrls')
    figure
    set(gcf,'Color','w');
    plot(time,mean_Env_index,'r','Displayname','D1 desync peak');
    hold on
    plot(time,mean_Env_pinky,'b','Displayname','D4 desync peak');
    legend
    
    %% compare peaks
    fg = figure;
    ft_plot_mesh(meshes,'facecolor',[.5 .5 .5],'facealpha',.1,'edgecolor','none')
    hold on
    axis equal
    view([-120 30])
    fig = gcf;
    fig.Color = 'w';
    ax = gca;
    ax.FontSize = 14;
%     ax.Children(3).HandleVisibility = 'off';
%     ax.Children(2).HandleVisibility = 'off';
%     ax.Children.HandleVisibility = 'off';
    % Find voxel with min/max value of Tstat
    h_index = plot3(dip_loc_D1(1),dip_loc_D1(2),dip_loc_D1(3),'x',...
        'markersize',10,'LineWidth',3,'DisplayName','D1');
    h_pinky = plot3(dip_loc_D4(1),dip_loc_D4(2),dip_loc_D4(3),'x',...
        'markersize',10,'LineWidth',3,'DisplayName','D4');
    legend
    [k2,~] = convhull(sourcepos,'Simplify',true);
    trisurf(k2,sourcepos(:,1),sourcepos(:,2),sourcepos(:,3),...
        'FaceColor',[0.5,0.5,0.5],'FaceAlpha',0.5,'EdgeColor','none',...
        'DisplayName','Sensorimotor VOI')
    dip_loc_D1_mm = dip_loc_D1.*1000;
    dip_loc_D4_mm = dip_loc_D4.*1000;
    save([path_Tstat,'D1_peak.txt'],'dip_loc_D1_mm','-ascii','-double')
    save([path_Tstat,'D4_peak.txt'],'dip_loc_D4_mm','-ascii','-double')
    %% save Tstat maps
    image = downsample_brain; 
    image.anatomy(:) = 0;
    image.anatomy(sub2ind(image.dim,...
        sourcepos_vox(:,1),sourcepos_vox(:,2),sourcepos_vox(:,3))) = pseudoT_D1;
    ft_write_mri([path_Tstat,files_Tstat,'index.nii'],image,...
        'dataformat','nifti');

    image.anatomy = image.anatomy./abs(min(image.anatomy));
    ft_write_mri([path_Tstat,files_Tstat,'index_norm.nii'],image,...
        'dataformat','nifti');
    
    image.anatomy(:) = 0;
    image.anatomy(sub2ind(image.dim,...
        sourcepos_vox(:,1),sourcepos_vox(:,2),sourcepos_vox(:,3))) = pseudoT_D4;
    ft_write_mri([path_Tstat,files_Tstat,'pinky.nii'],image,...
        'dataformat','nifti');

    image.anatomy = image.anatomy./abs(min(image.anatomy));
    ft_write_mri([path_Tstat,files_Tstat,'pinky_norm.nii'],image,...
        'dataformat','nifti');
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%

end
end

% end

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

function pseudoT = get_pseudoT(C,Ca,Cc,Lead_fields,mu)
% Calculate pseudo T statistic
% mu = 0.05;
Cr = C + mu*max(svd(C))*eye(size(C));
Cr_inv = inv(Cr);
pseudoT = zeros(1,size(Lead_fields,3));
infolength = 0;
for n = 1:size(Lead_fields,3)
    this_L = Lead_fields(:,:,n);
    W_v = inv((this_L'*Cr_inv*this_L))*(this_L'*Cr_inv);
    iPower_v = this_L'*Cr_inv*this_L;
    [v,d] = svd(iPower_v);
    [~,id] = min(diag(d));
    lopt = this_L*v(:,id); % turn to nAm amplitude
    w = (lopt'*Cr_inv/(lopt'*Cr_inv*lopt))';
    pseudoT(:,n) = (w'*Ca*w - w'*Cc*w)./(0.5.*(w'*Ca*w + w'*Cc*w));
    
    fprintf(repmat('\b',1,infolength));
    infolength = fprintf('Evaluating pseudo-T (%d/%d)\n',n,size(Lead_fields,3));
end
end

function [mean_Env] = get_envelope(trial_data,C,Lead_fields,min_voxel_index,control_inds,mu)
% mu = 0.05;
Cr = C + mu*max(svd(C))*eye(size(C));
Cr_inv = inv(Cr);

this_L = Lead_fields(:,:,min_voxel_index);
W_v = inv((this_L'*Cr_inv*this_L))*(this_L'*Cr_inv);
iPower_v = this_L'*Cr_inv*this_L;
[v,d] = svd(iPower_v);
[~,id] = min(diag(d));
lopt = this_L*v(:,id); % turn to nAm amplitude
w = (lopt'*Cr_inv/(lopt'*Cr_inv*lopt))';
N_trials = size(trial_data,3);
VEs = zeros(size(trial_data,2),N_trials);
Envs = VEs;
for tr_i = 1:N_trials
    VEs(:,tr_i) = w'*trial_data(:,:,tr_i) ./sqrt(w'*w);
    Envs(:,tr_i) = abs(hilbert(VEs(:,tr_i)));
    Envs(:,tr_i) = (Envs(:,tr_i) - mean(Envs(control_inds,tr_i)))./mean(Envs(control_inds,tr_i));
end
mean_Env = mean(Envs,2);
end