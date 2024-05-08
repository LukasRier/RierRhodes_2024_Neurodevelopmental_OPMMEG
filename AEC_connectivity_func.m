function AEC_connectivity_func(sub,ses,project_dir)
restoredefaultpath
cleaning_only = 0;
fake_bad_ch =1;
close all
clc
% project_dir = 'P:\Paediatric_OPM_Notts\';
% sub = '006';
% ses = '001';
run = 'run-001'
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
files_VEs = [filename,'_VE'];

path_AEC = [datadir,'derivatives',filesep,'AEC',filesep,'sub-',sub,filesep];
if ~exist(path_AEC,'dir'); mkdir(path_AEC);end
files_AEC = [filename,'_AEC'];

path_meg_data = [path_main,'meg',filesep];

path_meshes = [datadir,'derivatives',filesep,'sourcespace',filesep,'sub-',sub,filesep];
files_meshes = ['sub-',sub,'_meshes.mat'];
files_AAL_centroids = ['sub-',sub,'_AAL_centroids.nii.gz'];
files_AAL_regions = ['sub-',sub,'_AAL_regions.nii.gz'];

path_mri = [path_main,'anat',filesep];
files_mri = ['sub-',sub,'_anat.nii'];
S.mri_file = [path_mri,files_mri]; 

path_helmet = [datadir,'derivatives',filesep,'helmet',filesep,'sub-',sub,filesep];
files_helmet_info = dir([path_helmet,'*.mat']);files_helmet_info=files_helmet_info.name;

files_voxlox = ['sub-',sub,'_AAL_locs.mat'];
files_channels = [filename,'_channels.tsv'];

files_events = [filename,'_events.tsv'];


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

%% getting slot numbers for later. Change ICA comp visualiser?
precision_limit = 1e-6;
pos = [ch_table.Px,ch_table.Py,ch_table.Pz];
for sl_i = 1:size(Helmet_info.lay.pos,1)
    detected_ind = find(sqrt(sum((repmat(Helmet_info.sens_pos(sl_i,:),height(ch_table),1) - pos/100).^2,2)) < precision_limit)
    if ~isempty(detected_ind)
        ch_table.slot_no(detected_ind) = sl_i;
    end
end
%%

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
AAL_locs_mat_file = [path_meshes,files_AAL_centroids(1:end-7),'.mat'];
if ~exist(AAL_locs_mat_file,'file')
    AAL_regions = ft_read_mri([path_meshes,files_AAL_regions]);
    AAL_regions = ft_convert_units(AAL_regions,'m');
    [sourcepos_vox] = get_AAL_coords(AAL_regions,S)
    sourcepos = ft_warp_apply(AAL_regions.transform,sourcepos_vox);
    save(AAL_locs_mat_file,'sourcepos')
else
    load(AAL_locs_mat_file,'sourcepos')
end
[bf_outs_shell] = run_beamformer('shell',sourcepos,S,0,[],1);
lead_fields_shell_xyz = bf_outs_shell.LF;
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
scatter3(sourcepos(16,1),sourcepos(16,2),sourcepos(16,3),'r','linewidth',4)
scatter3(sourcepos(:,1),sourcepos(:,2),sourcepos(:,3),'r','linewidth',4)
% quiver3(S.sensor_info.pos(ch_table.isx==1,1),S.sensor_info.pos(ch_table.isx==1,2),S.sensor_info.pos(ch_table.isx==1,3),...
%     S.sensor_info.ors(ch_table.isx==1,1).*Lead_fields(ch_table.isx==1,2,16) + S.sensor_info.ors(ch_table.isy==1,1).*Lead_fields(ch_table.isy==1,2,16) + S.sensor_info.ors(ch_table.isz==1,1).*Lead_fields(ch_table.isz==1,2,16),...
%     S.sensor_info.ors(ch_table.isx==1,2).*Lead_fields(ch_table.isx==1,2,16) + S.sensor_info.ors(ch_table.isy==1,2).*Lead_fields(ch_table.isy==1,2,16) + S.sensor_info.ors(ch_table.isz==1,2).*Lead_fields(ch_table.isz==1,2,16),...
%     S.sensor_info.ors(ch_table.isx==1,3).*Lead_fields(ch_table.isx==1,2,16) + S.sensor_info.ors(ch_table.isy==1,3).*Lead_fields(ch_table.isy==1,2,16) + S.sensor_info.ors(ch_table.isz==1,3).*Lead_fields(ch_table.isz==1,2,16),'r','linewidth',2)
quiver3(S.sensor_info.pos(ch_table.isz==1,1),S.sensor_info.pos(ch_table.isz==1,2),S.sensor_info.pos(ch_table.isz==1,3),...
    S.sensor_info.ors(ch_table.isz==1,1).*Lead_fields(ch_table.isz==1,2,16),...
    S.sensor_info.ors(ch_table.isz==1,2).*Lead_fields(ch_table.isz==1,2,16),...
    S.sensor_info.ors(ch_table.isz==1,3).*Lead_fields(ch_table.isz==1,2,16),'r','linewidth',2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except fs epoch_length path_* files_* sub ses epoch_length data_f_clean cleaning_only fs sourcepos Lead_fields N_clean_trls hp lp
%%
%% filter the OPM data to band of interest
if ~cleaning_only
    %%% broadband VEs for spectra
    if ~exist(sprintf('%s%s_%d_%d_Hz_Z.mat',path_VEs,files_VEs,hp,lp),'file')
        
        data_f = data_f_clean';
        duration = epoch_length*N_clean_trls;
        
        %% Beamform
        C = cov(data_f);
        mu = 0.05;
        Cr = C + mu*max(svd(C))*eye(size(C));
        Cr_inv = inv(Cr);
        for n = 1:length(sourcepos)
            this_L = Lead_fields(:,:,n);
            W_v = inv((this_L'*Cr_inv*this_L))*(this_L'*Cr_inv);
            iPower_v = this_L'*Cr_inv*this_L;
            [v,d] = svd(iPower_v);
            [~,id] = min(diag(d));
            lopt = this_L*v(:,id); % turn to nAm amplitude
            w = (lopt'*Cr_inv/(lopt'*Cr_inv*lopt));
            VE(:,n) = (w*data_f')./sqrt(w*w');
        end
        save(sprintf('%s%s_%d_%d_Hz_Z.mat',path_VEs,files_VEs,hp,lp),'VE')
    end
    
    
    %%% narrow band VE and connectivity 
    hpfs = [13];
    lpfs = [30];
    
    for f_i = 1:length(hpfs)
       
        % hp = 13;
        % lp = 30;
        hp = hpfs(f_i);
        lp = lpfs(f_i);
        if ~exist(sprintf('%s%s_%d_%d_Hz_Z.mat',path_AEC,files_AEC,hp,lp),'file')
        [b,a] = butter(4,2*[hp lp]/fs);
        data_f = [filtfilt(b,a,data_f_clean')];
        duration = epoch_length*N_clean_trls;
        VE=[];
        %% Beamform
        C = cov(data_f);
        mu = 0.05;
        Cr = C + mu*max(svd(C))*eye(size(C));
        Cr_inv = inv(Cr);
        for n = 1:length(sourcepos)
            this_L = Lead_fields(:,:,n);
            W_v = inv((this_L'*Cr_inv*this_L))*(this_L'*Cr_inv);
            iPower_v = this_L'*Cr_inv*this_L;
            [v,d] = svd(iPower_v);
            [~,id] = min(diag(d));
            lopt = this_L*v(:,id); % turn to nAm amplitude
            w = (lopt'*Cr_inv/(lopt'*Cr_inv*lopt));
            VE(:,n) = (w*data_f')./sqrt(w*w');
        end
        save(sprintf('%s%s_%d_%d_Hz_Z.mat',path_VEs,files_VEs,hp,lp),'VE')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%
%         only take rebound window
        VE_mat = reshape(VE,[],epoch_length*fs,length(sourcepos));
        size(VE_mat)
%         win = [0.2,1.5];
        win = [0.1 3.4];
        VE_mat_rebound = VE_mat(:,round(win(1)*fs):round(win(2)*fs)-1,:);
        VE = reshape(VE_mat_rebound,[],length(sourcepos));
        duration = diff(win)*size(VE_mat,1);
        %%
        down_f = 10;
        Nlocs = 78;
        AEC = zeros(Nlocs,Nlocs);
        for seed = 1:Nlocs
            Xsig = squeeze(VE(:,seed));
            parfor test = 1:Nlocs
                if seed == test
                    AEC(seed,test) = NaN;
                else
                    Ysig = squeeze(VE(:,test));
                    X_win = (Xsig - mean(Xsig));
                    Y_win = (Ysig - mean(Ysig));
                    %%regress leakage
                    beta_leak = (pinv(X_win)*Y_win);
                    Y_win_cor = Y_win - X_win*beta_leak;
                    %%calculate envelopes
                    H_X = abs(hilbert(X_win));
                    H_X_d = mean(reshape(H_X,fs/down_f,round(duration*down_f),1));
                    %%calculate envelopes
                    H_Y = abs(hilbert(Y_win_cor));
                    H_Y_d = mean(reshape(H_Y,fs/down_f,round(duration*down_f),1));
                    AEC(seed,test) = corr(H_X_d',H_Y_d');
                end
            end
            if seed >1;fprintf(repmat('\b',1,72));end
            fprintf('Sub: %s | Session: %s | Freq %2d/%2d (%3d-%3d Hz) | AEC conn. region %2d\n',sub,ses,f_i,length(hpfs),hp,lp,seed);
        end
        AEC = 0.5*(AEC + AEC');
        figure()
        subplot(121)
        imagesc(AEC);colorbar;
        subplot(122)
        go_netviewer_perctl(AEC,0.95)
        drawnow
        
        AEC_seed = nan(size(AEC));
        seed_reg = 16; % 16 = postcentral/primary somatosensory cortex
        AEC_seed(seed_reg,:) =  AEC(seed_reg,:);
        AEC_seed(:,seed_reg) =  AEC(:,seed_reg);
        figure()
        set(gcf,'Position',[100,600,1200,400])
        subplot(131)
        go_netviewer_perctl(AEC_seed,0.5)
        view([0,0])
        subplot(132)
        go_netviewer_perctl(AEC_seed,0.5)
        subplot(133)
        go_netviewer_perctl(AEC_seed,0.5)
        view([-90,0])
        drawnow
        save(sprintf('%s%s_%d_%d_Hz_Z.mat',path_AEC,files_AEC,hp,lp),'AEC')
        end
    end
end
end

function trl = sampleinfo2trl(data)
% borrowed from private FieldTrip funcs
% SAMPLEINFO2TRL constructs the trial definition from the sampleinfo, the time axes
% and optionally from the trialinfo
%
% Use as
%   trl = sampleinfo2trl(data)
%
% See also ARTIFACT2BOOLVEC, ARTIFACT2EVENT, ARTIFACT2TRL, BOOLVEC2ARTIFACT, BOOLVEC2EVENT, BOOLVEC2TRL, EVENT2ARTIFACT, EVENT2BOOLVEC, EVENT2TRL, TRL2ARTIFACT, TRL2BOOLVEC, TRL2EVENT

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