clear all
close all
clc
overwrite = false;
project_dir =  '.\Adults\';

if overwrite
for sub_i = [1:19,21:22,24:26]
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    peak_VE_hmm_func(sub,ses,project_dir);
    %     input('next?')
    drawnow
end
%
project_dir =  '.\Children\';

for sub_i = 1:27
    sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    peak_VE_hmm_func(sub,ses,project_dir);
    %     input('next?')
    drawnow
end



% %
% project_dir =  '.\Adults_badtrls\';
% subs = [1:26];
% for sub_i = subs
% 
%     sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
%     ses_i = 1;
%     ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
%     peak_VE_hmm_func(sub,ses,project_dir);
%         input('next?')
%     drawnow
% end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% burst amplitudes

project_dir =  '.\Adults\';
for sub_i = [1:19,21:22,24:26]
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    [burst_amp_D1,...
        burst_amp_D4,...
        burst_amp_D1_std,...
        burst_amp_D4_std] = peak_VE_hmm_burst_amp(sub,ses,project_dir);
    drawnow
end

project_dir =  '.\Children\';
for sub_i = 1:27
    sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0'
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    [burst_amp_D1,...
        burst_amp_D4,...
        burst_amp_D1_std,...
        burst_amp_D4_std] = peak_VE_hmm_burst_amp(sub,ses,project_dir);
    drawnow
end

% project_dir =  '.\Adults_badtrls\';
% subs = [1:26];
% for sub_i = subs
% 
%     sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
%     ses_i = 1;
%     ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
%     [burst_amp_D1,...
%         burst_amp_D4,...
%         burst_amp_D1_std,...
%         burst_amp_D4_std] = peak_VE_hmm_burst_amp(sub,ses,project_dir);
%     drawnow
% end
end

%%
clearvars

% hmm_results_adults = table({},[],[],[],[],[],...
%     'VariableNames',{'ID','burst_frequency','burst_duration',...
%     'burst_interval','burst_raster','burst_spec'});
% hmm_results_kids = table({},[],[],[],[],[],[],...
%     'VariableNames',{'ID','burst_frequency','burst_duration',...
%     'burst_interval','burst_raster','burst_spec','age'});
hmm_results_adults = table({},...
    'VariableNames',{'ID'});
hmm_results_kids = table({},...
    'VariableNames',{'ID'});

close all
burst_frequency = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);
burst_duration = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);
burst_amplitude = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);
burst_interval = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);
burst_raster = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);
burst_raster_all = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);
burst_spec = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);
non_burst_spec = struct('Adults_D1',[],'Adults_D4',[],'Children_D1',[],'Children_D4',[]);

project_dir =  '.\Adults\';
load(['.\sub_info_adults.mat'],'AdultDatasets')
% project_dir =  '.\Adults_badtrls\';

AdultDatasets.Age = years(AdultDatasets.Date - AdultDatasets.DOB)
adults_ages = [];
for sub_i = [1:19,21:22,24:26]

    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    hmm_results_adults.ID(height(hmm_results_adults)+1) = {sub};
    adults_ages = cat(1,adults_ages,AdultDatasets.Age(startsWith(AdultDatasets.SubjID,sub)));

    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    run = 'run-001';
    
    datadir = [project_dir,'Data',filesep,'BIDS',filesep];
    exp_type = '_task-braille';
    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];

    path_hmm = [datadir,'derivatives',filesep,'HMM',filesep,'sub-',sub,filesep];
    file_hmm = [filename,'_hmm-outputs.mat'];
    hmm_outs = load([path_hmm,file_hmm]);

    burst_frequency.Adults_D1 = [burst_frequency.Adults_D1;hmm_outs.num_bursts_D1];
    burst_frequency.Adults_D4 = [burst_frequency.Adults_D4;hmm_outs.num_bursts_D4];
    
    burst_duration.Adults_D1 = [burst_duration.Adults_D1;hmm_outs.burst_dur_D1];
    burst_duration.Adults_D4 = [burst_duration.Adults_D4;hmm_outs.burst_dur_D4];

    burst_interval.Adults_D1 = [burst_interval.Adults_D1;hmm_outs.burst_ISL_D1];
    burst_interval.Adults_D4 = [burst_interval.Adults_D4;hmm_outs.burst_ISL_D4];
    
    [burst_amp_D1,...
        burst_amp_D4,...
        burst_amp_D1_std,...
        burst_amp_D4_std] = peak_VE_hmm_burst_amp(sub,ses,project_dir);
    burst_amplitude.Adults_D1 = [burst_amplitude.Adults_D1;burst_amp_D1];
    burst_amplitude.Adults_D4 = [burst_amplitude.Adults_D4;burst_amp_D4];

    rasters_D1 = reshape(hmm_outs.burst_mask_D1,size(hmm_outs.state_probabilities_D1,1),[])';
    rasters_D4 = reshape(hmm_outs.burst_mask_D4,size(hmm_outs.state_probabilities_D4,1),[])';
    
    burst_raster.Adults_D1 = [burst_raster.Adults_D1;rasters_D1];
    burst_raster.Adults_D4 = [burst_raster.Adults_D4;rasters_D4];
    burst_raster_all.Adults_D1 = cat(1,burst_raster_all.Adults_D1,{rasters_D1});
    burst_raster_all.Adults_D4 = cat(1,burst_raster_all.Adults_D4,{rasters_D4});

    burst_spec.Adults_D1 = [burst_spec.Adults_D1, hmm_outs.fit_D1.state(hmm_outs.burst_state_D1).psd];
    burst_spec.Adults_D4 = [burst_spec.Adults_D4, hmm_outs.fit_D4.state(hmm_outs.burst_state_D4).psd];
    
    non_b_states_D1 = true(1,3);non_b_states_D1(hmm_outs.burst_state_D1) = false;
    non_b_states_D4 = true(1,3);non_b_states_D4(hmm_outs.burst_state_D4) = false;
    non_burst_spec.Adults_D1 = [non_burst_spec.Adults_D1, mean([hmm_outs.fit_D1.state(non_b_states_D1).psd],2)];
    non_burst_spec.Adults_D4 = [non_burst_spec.Adults_D4, mean([hmm_outs.fit_D1.state(non_b_states_D4).psd],2)];

end

hmm_results_adults.burst_frequency_D1 =  burst_frequency.Adults_D1;
hmm_results_adults.burst_frequency_D4 =  burst_frequency.Adults_D4;
hmm_results_adults.burst_duration_D1 = burst_duration.Adults_D1;
hmm_results_adults.burst_duration_D4 = burst_duration.Adults_D4;
hmm_results_adults.burst_amplitude_D1 = burst_amplitude.Adults_D1;
hmm_results_adults.burst_amplitude_D4 = burst_amplitude.Adults_D4;
hmm_results_adults.burst_interval_D1 = burst_interval.Adults_D1;
hmm_results_adults.burst_interval_D4 = burst_interval.Adults_D4;
hmm_results_adults.burst_spec_D1 = burst_spec.Adults_D1';
hmm_results_adults.burst_spec_D4 = burst_spec.Adults_D4';
hmm_results_adults.non_burst_spec_D1 = non_burst_spec.Adults_D1';
hmm_results_adults.non_burst_spec_D4 = non_burst_spec.Adults_D4';
hmm_results_adults.burst_raster_D1 = burst_raster_all.Adults_D1;
hmm_results_adults.burst_raster_D4 = burst_raster_all.Adults_D4;

hmm_results_adults.age = adults_ages;

%%
project_dir =  '.\Children\';
load(['sub_info.mat'],'KidsDatasets')
KidsDatasets.Age_yrs = years(KidsDatasets.Date - KidsDatasets.DOB);
kids_ages = [];

for sub_i = 1:27
    sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0'
    hmm_results_kids.ID(height(hmm_results_kids)+1) = {sub};
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0'
    run = 'run-001';
    kids_ages = cat(1,kids_ages,KidsDatasets.Age_yrs(startsWith(KidsDatasets.SubjID,sub)));

    datadir = [project_dir,'Data',filesep,'BIDS',filesep];
    exp_type = '_task-braille';
    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];
    
    path_hmm = [datadir,'derivatives',filesep,'HMM',filesep,'sub-',sub,filesep];
    file_hmm = [filename,'_hmm-outputs.mat'];
    hmm_outs = load([path_hmm,file_hmm]);

    burst_frequency.Children_D1 = [burst_frequency.Children_D1;hmm_outs.num_bursts_D1];
    burst_frequency.Children_D4 = [burst_frequency.Children_D4;hmm_outs.num_bursts_D4];
    
    burst_duration.Children_D1 = [burst_duration.Children_D1;hmm_outs.burst_dur_D1];
    burst_duration.Children_D4 = [burst_duration.Children_D4;hmm_outs.burst_dur_D4];

    burst_interval.Children_D1 = [burst_interval.Children_D1;hmm_outs.burst_ISL_D1];
    burst_interval.Children_D4 = [burst_interval.Children_D4;hmm_outs.burst_ISL_D4];
        [burst_amp_D1,...
        burst_amp_D4,...
        burst_amp_D1_std,...
        burst_amp_D4_std] = peak_VE_hmm_burst_amp(sub,ses,project_dir);
    burst_amplitude.Children_D1 = [burst_amplitude.Children_D1;burst_amp_D1];
    burst_amplitude.Children_D4 = [burst_amplitude.Children_D4;burst_amp_D4];

    rasters_D1 = reshape(hmm_outs.burst_mask_D1,size(hmm_outs.state_probabilities_D1,1),[])';
    rasters_D4 = reshape(hmm_outs.burst_mask_D4,size(hmm_outs.state_probabilities_D4,1),[])';
    
    burst_raster.Children_D1 = [burst_raster.Children_D1;rasters_D1];
    burst_raster.Children_D4 = [burst_raster.Children_D4;rasters_D4];
    burst_raster_all.Children_D1 = cat(1,burst_raster_all.Children_D1,{rasters_D1});
    burst_raster_all.Children_D4 = cat(1,burst_raster_all.Children_D4,{rasters_D4});

    burst_spec.Children_D1 = [burst_spec.Children_D1, hmm_outs.fit_D1.state(hmm_outs.burst_state_D1).psd];
    burst_spec.Children_D4 = [burst_spec.Children_D4, hmm_outs.fit_D4.state(hmm_outs.burst_state_D4).psd];

    non_b_states_D1 = true(1,3);non_b_states_D1(hmm_outs.burst_state_D1) = false;
    non_b_states_D4 = true(1,3);non_b_states_D4(hmm_outs.burst_state_D4) = false;
    non_burst_spec.Children_D1 = [non_burst_spec.Children_D1, mean([hmm_outs.fit_D1.state(non_b_states_D1).psd],2)];
    non_burst_spec.Children_D4 = [non_burst_spec.Children_D4, mean([hmm_outs.fit_D1.state(non_b_states_D4).psd],2)];

end


hmm_results_kids.burst_frequency_D1 =  burst_frequency.Children_D1;
hmm_results_kids.burst_frequency_D4 =  burst_frequency.Children_D4;
hmm_results_kids.burst_duration_D1 = burst_duration.Children_D1;
hmm_results_kids.burst_duration_D4 = burst_duration.Children_D4;
hmm_results_kids.burst_amplitude_D1 = burst_amplitude.Children_D1;
hmm_results_kids.burst_amplitude_D4 = burst_amplitude.Children_D4;
hmm_results_kids.burst_interval_D1 = burst_interval.Children_D1;
hmm_results_kids.burst_interval_D4 = burst_interval.Children_D4;
hmm_results_kids.burst_spec_D1 = burst_spec.Children_D1';
hmm_results_kids.burst_spec_D4 = burst_spec.Children_D4';
hmm_results_kids.non_burst_spec_D1 = non_burst_spec.Children_D1';
hmm_results_kids.non_burst_spec_D4 = non_burst_spec.Children_D4';

hmm_results_kids.burst_raster_D1 = burst_raster_all.Children_D1;
hmm_results_kids.burst_raster_D4 = burst_raster_all.Children_D4;
hmm_results_kids.age = kids_ages;
% save('HMM_RESULTS_badtrl','hmm_results_kids','hmm_results_adults')
save('HMM_RESULTS','hmm_results_kids','hmm_results_adults')
%%
addpath .\Violinplot-Matlab-master\
burst_frequency.Children_D1 = hmm_results_kids.burst_frequency_D1;
burst_frequency.Children_D4 = hmm_results_kids.burst_frequency_D4;
burst_duration.Children_D1 = hmm_results_kids.burst_duration_D1;
burst_duration.Children_D4 = hmm_results_kids.burst_duration_D4;
burst_interval.Children_D1 = hmm_results_kids.burst_interval_D1;
burst_interval.Children_D4 = hmm_results_kids.burst_interval_D4;


figure  
vs = violinplot(burst_duration);
[vs.ShowMean] = deal(1);
[vs.ShowBox] = deal(0);
title('burst duration (ms)')

figure  
vs = violinplot(burst_interval);
[vs.ShowMean] = deal(1);
[vs.ShowBox] = deal(0);
title('inter burst interval (ms)')

figure  
vs = violinplot(burst_frequency);
[vs.ShowMean] = deal(1);
[vs.ShowBox] = deal(0);
title('burst frequency s^-1')

figure  
vs = violinplot(burst_amplitude);
[vs.ShowMean] = deal(1);
[vs.ShowBox] = deal(0);
title('burst amplitude')
[p,H,stats] = ranksum(burst_amplitude.Adults_D1,burst_amplitude.Children_D4)
[p,H,stats] = ranksum(burst_amplitude.Adults_D4,burst_amplitude.Children_D4)

figure
set(gcf,'Color','w','Position', [680 71 842 907])
tiledlayout(1,2)
nexttile
imagesc(burst_raster.Adults_D1)
title('Adults_{D1}')
ylabel('trial #')
xlabel('time (s)')
xticklabels(strsplit(num2str(xticks./100)))

% nexttile
% imagesc(burst_raster.Adults_D4)
% title('Adults_{D4}')
% ylabel('trial #')
% xlabel('time (s)')
% xticklabels(strsplit(num2str(xticks./100)))

nexttile
imagesc(burst_raster.Children_D1)
title('Children_{D1}')
ylabel('trial #')
xlabel('time (s)')
xticklabels(strsplit(num2str(xticks./100)))

% nexttile
% imagesc(burst_raster.Children_D4)
% title('Children_{D4}')
% ylabel('trial #')
% xlabel('time (s)')
% xticklabels(strsplit(num2str(xticks./100)))

figure
plot(mean(burst_raster.Adults_D1,1),'DisplayName','Adults_{D1}')
hold on
plot(mean(burst_raster.Adults_D4,1),'DisplayName','Adults_{D4}')
plot(mean(burst_raster.Children_D1,1),'DisplayName','Children_{D1}')
plot(mean(burst_raster.Children_D4,1),'DisplayName','Children_{D4}')
legend
set(gcf,'Color','w')
ylabel('Burst probability')
cols = lines;
%% quartile range 
figure
h=[];
f = hmm_outs.fit_D1.state(1).f;
m_Ch_D1 = median(burst_spec.Children_D1,2);
m_Ch_D4 = median(burst_spec.Children_D4,2);
qrtls_Ch_D1 = prctile(burst_spec.Children_D1, [25, 75, 50],2);
qrtls_Ch_D4 = prctile(burst_spec.Children_D4, [25, 75, 50],2);
% m_Ch_D1 = qrtls_Ch_D1(:,3);
% m_Ch_D4 = qrtls_Ch_D4(:,3);

m_Ad_D1 = median(burst_spec.Adults_D1,2);
m_Ad_D4 = median(burst_spec.Adults_D4,2);
qrtls_Ad_D1 = prctile(burst_spec.Adults_D1, [25, 75, 50],2);
qrtls_Ad_D4 = prctile(burst_spec.Adults_D4, [25, 75, 50],2);
% m_Ad_D1 = qrtls_Ad_D1(:,3);
% m_Ad_D4 = qrtls_Ad_D4(:,3);

subplot(1,2,1)
title('D1')
hold on
plot(f,m_Ch_D1,'LineWidth',2,'DisplayName','Children')
h(end+1) = ciplot(qrtls_Ch_D1(:,1),qrtls_Ch_D1(:,2),f,cols(1,:));

plot(f,m_Ad_D1,'Color',cols(2,:),'LineWidth',2,'DisplayName','Adults')
h(end+1) = ciplot(qrtls_Ad_D1(:,1),qrtls_Ad_D1(:,2),f,cols(2,:));
legend
ylims=[0,0.08];
ylim(ylims)


subplot(1,2,2)
title('D4')
hold on
plot(f,m_Ch_D4,'LineWidth',2)
plot(f,m_Ad_D4,'LineWidth',2)
h(end+1) = ciplot(qrtls_Ch_D4(:,1),qrtls_Ch_D4(:,2),f,cols(1,:));

h(end+1) = ciplot(qrtls_Ad_D4(:,1),qrtls_Ad_D4(:,2),f,cols(2,:));
ylim(ylims)
set(h(1:2),'DisplayName','IQR')

set(h(3:end),'HandleVisibility','off')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
title('D1')
hold on
h1=[];
plot(f,m_Ch_D1,'LineWidth',2,'DisplayName','Children')
h1(end+1) = ciplot(qrtls_Ch_D1(:,1),qrtls_Ch_D1(:,2),f,cols(1,:));

plot(f,m_Ad_D1,'Color',cols(2,:),'LineWidth',2,'DisplayName','Adults')
h1(end+1) = ciplot(qrtls_Ad_D1(:,1),qrtls_Ad_D1(:,2),f,cols(2,:));
legend
ylims=[0,0.08];
ylim(ylims)
set(h1(1:2),'DisplayName','IQR')

set(h1(3:end),'HandleVisibility','off')
set(gcf,'Color','w')
xlabel('frequency (Hz)')
ylabel('Burst PSD (AU)')
%%%%%%%%%%%%%%%%%%%%%%
%% mean and std errs
figure
f = hmm_outs.fit_D1.state(1).f;
m_Ch_D1 = mean(burst_spec.Children_D1,2);
m_Ch_D4 = mean(burst_spec.Children_D4,2);
st_Ch_D1 = std(burst_spec.Children_D1,[],2)./sqrt(size(burst_spec.Children_D1,2));
st_Ch_D4 = std(burst_spec.Children_D4,[],2)./sqrt(size(burst_spec.Children_D4,2));

m_Ad_D1 = mean(burst_spec.Adults_D1,2);
m_Ad_D4 = mean(burst_spec.Adults_D4,2);
st_Ad_D1 = std(burst_spec.Adults_D1,[],2)./sqrt(size(burst_spec.Adults_D1,2));
st_Ad_D4 = std(burst_spec.Adults_D4,[],2)./sqrt(size(burst_spec.Adults_D4,2));
h=[];
subplot(1,2,1)
title('D1')
hold on
plot(f,m_Ch_D1,'LineWidth',2,'DisplayName','Children')
h(end+1) = ciplot(m_Ch_D1-st_Ch_D1,m_Ch_D1+st_Ch_D1,f,cols(1,:));

plot(f,m_Ad_D1,'Color',cols(2,:),'LineWidth',2,'DisplayName','Adults')
h(end+1) = ciplot(m_Ad_D1-st_Ad_D1,m_Ad_D1+st_Ad_D1,f,cols(2,:));
legend
ylims=[0,0.071];
ylim(ylims)
subplot(1,2,2)
title('D4')
hold on
plot(f,m_Ch_D4,'LineWidth',2)
plot(f,m_Ad_D4,'LineWidth',2)
h(end+1) = ciplot(m_Ch_D4-st_Ch_D4,m_Ch_D4+st_Ch_D4,f,cols(1,:));

h(end+1) = ciplot(m_Ad_D4-st_Ad_D4,m_Ad_D4+st_Ad_D4,f,cols(2,:));
ylim(ylims)
set(h(1:2),'DisplayName','std. error')

set(h(3:end),'HandleVisibility','off')

%%
figure

t = linspace(0,size(hmm_outs.state_probabilities_D1,1)./hmm_outs.options.Fs,...
    size(hmm_outs.state_probabilities_D1,1));

figure
set(gcf,'Color','w','Position', [680 71 842 907])
tiledlayout(2,2)
nexttile
ph = []; cb = []; ax = [];clims = [0,0.022];
pseudo_TFS_Ad_D1 = repmat(m_Ad_D1,1,size(hmm_outs.state_probabilities_D1,1)).* ...
    repmat(mean(burst_raster.Adults_D1,1),size(f,2),1);
ph(end+1) = pcolor(t,f, (pseudo_TFS_Ad_D1));
title('Adults_{D1}')
cb(end+1) = colorbar;
ax(end+1) = gca;
xlabel('time / s') ; ylabel('f / Hz');

nexttile
pseudo_TFS_Ad_D4 = repmat(m_Ad_D4,1,size(hmm_outs.state_probabilities_D1,1)).* ...
    repmat(mean(burst_raster.Adults_D4,1),size(f,2),1);
ph(end+1) = pcolor(t,f, (pseudo_TFS_Ad_D4));
title('Adults_{D4}')
cb(end+1) = colorbar;
ax(end+1) = gca;
xlabel('time / s') ; ylabel('f / Hz');

nexttile
pseudo_TFS_Ch_D1 = repmat(m_Ch_D1,1,size(hmm_outs.state_probabilities_D1,1)).* ...
    repmat(mean(burst_raster.Children_D1,1),size(f,2),1);
ph(end+1) = pcolor(t,f, (pseudo_TFS_Ch_D1));
title('Children_{D1}')
cb(end+1) = colorbar;
ax(end+1) = gca;
xlabel('time / s') ; ylabel('f / Hz');

nexttile
pseudo_TFS_Ch_D4 = repmat(m_Ch_D4,1,size(hmm_outs.state_probabilities_D1,1)).* ...
    repmat(mean(burst_raster.Children_D4,1),size(f,2),1);
ph(end+1) = pcolor(t,f, (pseudo_TFS_Ch_D4));
title('Children_{D4}')
cb(end+1) = colorbar;
ax(end+1) = gca;
xlabel('time / s') ; ylabel('f / Hz');
set(ph,'FaceColor','Interp','EdgeColor','none')
%%
figure
set(gcf,'Color','w','Position', [680 71 842 907])

pseudo_TFS_diff = ((pseudo_TFS_Ch_D1 + pseudo_TFS_Ch_D4)./2 - (pseudo_TFS_Ad_D1 + pseudo_TFS_Ad_D4)./2)./(pseudo_TFS_Ad_D1 + pseudo_TFS_Ad_D4)./2;


ph(end+1) = pcolor(t,f, (pseudo_TFS_diff));
title('Children - Adults pseudo TFS')
cb(end+1) = colorbar;colormap turbo
ax(end+1) = gca;
xlabel('time / s') ; ylabel('f / Hz');
set(ph(end),'FaceColor','Interp','EdgeColor','none')

%% age vs RMS spectral difference

m_Ch_D1 = mean(burst_spec.Children_D1,2);
m_Ch_D4 = mean(burst_spec.Children_D4,2);

m_Ad_D1 = mean(burst_spec.Adults_D1,2);
m_Ad_D4 = mean(burst_spec.Adults_D4,2);
RMS = @(mat,dim) sqrt(mean((mat).^2,dim));

RMS_Ad_D1 = RMS(burst_spec.Adults_D1 - m_Ad_D1,1);
RMS_Ad_D4 = RMS(burst_spec.Adults_D4 - m_Ad_D4,1);
RMS_Ch_D1 = RMS(burst_spec.Children_D1 - m_Ad_D1,1);
RMS_Ch_D4 = RMS(burst_spec.Children_D4 - m_Ad_D4,1);
%%
f_RMS = figure('Units','Centimeters');

subplot(1,2,1)
hold on
RMS_Ad_D1_std_dev = mean(RMS_Ad_D1);
scatter(kids_ages,RMS_Ch_D1)
yline(RMS_Ad_D1_std_dev,'-','Linewidth',2)
yline(RMS_Ad_D1_std_dev+std(RMS_Ad_D1)./length(RMS_Ad_D1),'--')
yline(RMS_Ad_D1_std_dev-std(RMS_Ad_D1)./length(RMS_Ad_D1),'--')

X = [ones(length(kids_ages),1) kids_ages];
b = X\RMS_Ch_D1';
RMS_Ch_D1_Calc2 = X*b;
plot(kids_ages,RMS_Ch_D1_Calc2,'r-')
ylabel('RMS difference with Adult mean spectrum')
xlabel('Age')


subplot(1,2,2)
RMS_Ad_D4_std_dev = mean(RMS_Ad_D4);


yline(RMS_Ad_D4_std_dev,'-','Linewidth',2)
yline(RMS_Ad_D4_std_dev+std(RMS_Ad_D4)./length(RMS_Ad_D4),'--')
yline(RMS_Ad_D4_std_dev-std(RMS_Ad_D4)./length(RMS_Ad_D4),'--')

hold on
scatter(kids_ages,RMS_Ch_D4)

X = [ones(length(kids_ages),1) kids_ages];
b = X\RMS_Ch_D4';
RMS_Ch_D4_Calc2 = X*b;
plot(kids_ages,RMS_Ch_D4_Calc2,'r-')
ylabel('RMS difference with Adult mean spectrum')
xlabel('Age')
[rho,pval] = corr(kids_ages,RMS_Ch_D4')

%% 3 age groups

[~,age_rank] = sort(kids_ages);

youngest = age_rank(1:floor(length(age_rank)/3));
oldest = age_rank(end - floor(length(age_rank)/3)+1 : end);
middle = age_rank(floor(length(age_rank)/3)+1:end - floor(length(age_rank)/3));
figure
plot(youngest,kids_ages(youngest),'*')
hold on
plot(middle,kids_ages(middle),'^')
plot(oldest,kids_ages(oldest),'s')

%% non burst spectra

%% quartile range 
figure
h=[];
f = hmm_outs.fit_D1.state(1).f;
non_m_Ch_D1 = median(non_burst_spec.Children_D1,2);
non_m_Ch_D4 = median(non_burst_spec.Children_D4,2);
non_qrtls_Ch_D1 = prctile(non_burst_spec.Children_D1, [25, 75, 50],2);
non_qrtls_Ch_D4 = prctile(non_burst_spec.Children_D4, [25, 75, 50],2);
% m_Ch_D1 = qrtls_Ch_D1(:,3);
% m_Ch_D4 = qrtls_Ch_D4(:,3);

non_m_Ad_D1 = median(non_burst_spec.Adults_D1,2);
non_m_Ad_D4 = median(non_burst_spec.Adults_D4,2);
non_qrtls_Ad_D1 = prctile(non_burst_spec.Adults_D1, [25, 75, 50],2);
non_qrtls_Ad_D4 = prctile(non_burst_spec.Adults_D4, [25, 75, 50],2);
% m_Ad_D1 = qrtls_Ad_D1(:,3);
% m_Ad_D4 = qrtls_Ad_D4(:,3);

subplot(1,2,1)
title('D1 non burst')
hold on
plot(f,non_m_Ch_D1,'LineWidth',2,'DisplayName','Children')
h(end+1) = ciplot(non_qrtls_Ch_D1(:,1),non_qrtls_Ch_D1(:,2),f,cols(1,:));

plot(f,non_m_Ad_D1,'Color',cols(2,:),'LineWidth',2,'DisplayName','Adults')
h(end+1) = ciplot(non_qrtls_Ad_D1(:,1),non_qrtls_Ad_D1(:,2),f,cols(2,:));
legend
ylims=[0,0.08];
ylim(ylims)


subplot(1,2,2)
title('D4 non burst')
hold on
plot(f,non_m_Ch_D4,'LineWidth',2)
plot(f,non_m_Ad_D4,'LineWidth',2)
h(end+1) = ciplot(non_qrtls_Ch_D4(:,1),non_qrtls_Ch_D4(:,2),f,cols(1,:));

h(end+1) = ciplot(non_qrtls_Ad_D4(:,1),non_qrtls_Ad_D4(:,2),f,cols(2,:));
ylim(ylims)
set(h(1:2),'DisplayName','IQR')

set(h(3:end),'HandleVisibility','off')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
title('D1')
hold on
h1=[];
plot(f,non_m_Ch_D1,'LineWidth',2,'DisplayName','Children')
h1(end+1) = ciplot(non_qrtls_Ch_D1(:,1),non_qrtls_Ch_D1(:,2),f,cols(1,:));

plot(f,non_m_Ad_D1,'Color',cols(2,:),'LineWidth',2,'DisplayName','Adults')
h1(end+1) = ciplot(non_qrtls_Ad_D1(:,1),non_qrtls_Ad_D1(:,2),f,cols(2,:));
legend
ylims=[0,0.08];
ylim(ylims)
set(h1(1:2),'DisplayName','IQR')

set(h1(3:end),'HandleVisibility','off')
set(gcf,'Color','w')
xlabel('frequency (Hz)')
ylabel('non Burst PSD (AU)')
%%%%%%%%%%%%%%%%%%%%%%

