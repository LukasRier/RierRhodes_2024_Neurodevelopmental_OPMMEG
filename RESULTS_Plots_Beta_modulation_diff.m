clearvars
close all
badtrls
if badtrls
    project_dir =  '.\Adults_badtrls\';
else
    project_dir =  '.\Adults\';

end
TFS_results_adults = table({},...
    'VariableNames',{'ID'});
TFS_results_kids = table({},...
    'VariableNames',{'ID'});
conwin = [2.5,3];
extension_T = '';

%% envelopes

mean_Env_index_all = [];
mean_Env_index_pinkytrls_all = [];
mean_Env_pinky_all = [];
mean_Env_pinky_indextrls_all = [];
% good_subs = 1:26;good_subs(good_subs==20|good_subs==23)=[];
good_subs = [1:19,21:22,24:26];
% good_subs = [1:26];
for sub_i = good_subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    TFS_results_adults.ID(height(TFS_results_adults)+1) = {sub};

    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';
    hp = 8;
    lp = 25;

    datadir = [project_dir,'Data',filesep,'BIDS',filesep];

    path_Tstat = [datadir,'derivatives',filesep,'Tstats',extension_T,filesep,'sub-',sub,filesep];

    %% envelopes
    load([path_Tstat,'D1_peak_env_13_30_Hz.mat'],'mean_Env_index','mean_Env_index_pinkytrls')
    load([path_Tstat,'D4_peak_env_13_30_Hz.mat'],'mean_Env_pinky','mean_Env_pinky_indextrls')
    %     load([path_Tstat,'D1_peak_env_8_25_Hz.mat'],'mean_Env_index','mean_Env_index_pinkytrls')
    %     load([path_Tstat,'D4_peak_env_8_25_Hz.mat'],'mean_Env_pinky','mean_Env_pinky_indextrls')

    mean_Env_index_all = cat(2,mean_Env_index_all,mean_Env_index);
    mean_Env_index_pinkytrls_all = cat(2,mean_Env_index_pinkytrls_all,mean_Env_index_pinkytrls);
    mean_Env_pinky_all = cat(2,mean_Env_pinky_all,mean_Env_pinky);
    mean_Env_pinky_indextrls_all = cat(2,mean_Env_pinky_indextrls_all,mean_Env_pinky_indextrls);
end

grp_mean_Env_index_all = mean(mean_Env_index_all,2);
grp_mean_Env_index_pinkytrls_all = mean(mean_Env_index_pinkytrls_all,2);
grp_mean_Env_pinky_all = mean(mean_Env_pinky_all,2);
grp_mean_Env_pinky_indextrls_all = mean(mean_Env_pinky_indextrls_all,2);

grp_std_Env_index_all = std(mean_Env_index_all,[],2);
grp_std_Env_index_pinkytrls_all = std(mean_Env_index_pinkytrls_all,[],2);
grp_std_Env_pinky_all = std(mean_Env_pinky_all,[],2);
grp_std_Env_pinky_indextrls_all = std(mean_Env_pinky_indextrls_all,[],2);
fs = 1200;
control_window = round(conwin.*fs);control_inds = control_window(1):control_window(2);

%%
figure
FtSz=14;
set(gcf,'Color','w')
set(gcf,'Position',[681 559 1120 420])

fs = 1200;
time = linspace(0,size(grp_mean_Env_index_all,1)./fs,size(grp_mean_Env_index_all,1));
hold on

ylims=[-0.4000 0.5484];
actwin = [0.3,0.8];
control_period = [conwin(1),ylims(1);conwin(1),ylims(2);conwin(2),ylims(2);...
    conwin(2),ylims(1);conwin(1),ylims(1)];
active_period = [actwin(1),ylims(1);actwin(1),ylims(2);actwin(2),ylims(2);...
    actwin(2),ylims(1);actwin(1),ylims(1)];

ph(1) = fill(control_period(:,1),control_period(:,2),[.5,.5,.5],'FaceAlpha',0.4,'DisplayName','Rest');
ph(2) = fill(active_period(:,1),active_period(:,2),'g','FaceAlpha',0.4,'DisplayName','Active');
set(ph,'EdgeColor','none','HandleVisibility','off')

plot(time,grp_mean_Env_index_all,'r','LineWidth',2,...
    'Displayname','D2 desync peak');
ciplot(grp_mean_Env_index_all-grp_std_Env_index_all,...
    grp_mean_Env_index_all+grp_std_Env_index_all,time,'r')

plot(time,grp_mean_Env_pinky_all,'b','LineWidth',2,...
    'Displayname','D5 desync peak');
ciplot(grp_mean_Env_pinky_all-grp_std_Env_pinky_all,...
    grp_mean_Env_pinky_all+grp_std_Env_pinky_all,time,'b')
lh = legend;lh.String{2} = 'std. dev.';lh.String{4} = 'std. dev.';
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;chldrn(3).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Fractional Change')


%%

f_D2_all = figure;f_D2_all.Color = 'w';set(f_D2_all,"Units","normalized","Position",[0,0,1,1])
TL_D2 = tiledlayout('flow','TileSpacing','Compact');
f_D5_all = figure;f_D5_all.Color = 'w';set(f_D5_all,"Units","normalized","Position",[0,0,1,1])
TL_D5 = tiledlayout('flow','TileSpacing','Compact');

TFS_all_D2= [];
TFS_all_D5 = [];
for sub_i = good_subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'

    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';

    datadir = [project_dir,'Data',filesep,'BIDS',filesep];

    path_Tstat = [datadir,'derivatives',filesep,'Tstats',extension_T,filesep,'sub-',sub,filesep];
    % load timecourses
    load([path_Tstat,'D1_peak_VE_1-150_Hz.mat'],'VE_index','VE_index_pinkytrls')
    load([path_Tstat,'D4_peak_VE_1-150_Hz.mat'],'VE_pinky','VE_pinky_indextrls')

    [TFS,fre] = VE_TFS(VE_index,control_inds,time,fs);
    TFS_all_D2 = cat(3,TFS_all_D2,TFS);

    figure(f_D2_all)
    nexttile
    pcolor(time,fre,TFS_all_D2(:,:,end));shading interp
    xlabel('Time (s)');ylabel('Frequency (Hz)')
    colorbar;cbar_lim = [0.5];
    caxis([-cbar_lim cbar_lim])
    axis fill
    title([sub])


    [TFS,fre] = VE_TFS(VE_pinky,control_inds,time,fs);
    TFS_all_D5 =  cat(3,TFS_all_D5,TFS);

    figure(f_D5_all)
    nexttile
    pcolor(time,fre,TFS_all_D5(:,:,end));shading interp
    xlabel('Time (s)');ylabel('Frequency (Hz)')
    colorbar;
    caxis([-cbar_lim cbar_lim])
    axis fill
    title([sub])
end
figure(f_D2_all)
sgtitle('D2')
figure(f_D5_all)
sgtitle('D5')
set(TL_D2.Children(2:2:end),'YLim',[5,50])
set(TL_D5.Children(2:2:end),'YLim',[5,50])

%%
FntS=15;
figure
set(gcf,'Color','w','Position',[681 510 900 400])
cbar_lim = [0.2];
subplot(1,2,1)
pcolor(time,fre,mean(TFS_all_D2,3));shading interp
xlabel('Time (s)');
ylabel('Frequency (Hz)')
cb = colorbar;
caxis([-cbar_lim cbar_lim])
% cb.Label.String = 'Fractional change';cb.Label.FontSize = FntS;
axis fill
ax=gca;
ax.FontSize=FntS;ax.YLabel.FontSize = FntS;
ax.XLabel.FontSize = FntS;
ylim([5,50])
title('D1')

subplot(1,2,2)
pcolor(time,fre,mean(TFS_all_D5,3));shading interp
xlabel('Time (s)');%ylabel('Frequency (Hz)')
cb = colorbar;caxis([-cbar_lim cbar_lim])
cb.Label.String = 'Fractional change';cb.Label.FontSize = FntS;
axis fill
set(gcf,'Name','D4 TFS')
ax=gca;
ax.FontSize=FntS;ax.YLabel.FontSize = FntS;
ax.XLabel.FontSize = FntS;
ylim([5,50])
title('D5')
sgtitle('Adults')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kids
project_dir =  '.\Children\';

extension_T = '';

%% envelopes

mean_Env_index_all_kids = [];
mean_Env_index_pinkytrls_all_kids = [];
mean_Env_pinky_all_kids = [];
mean_Env_pinky_indextrls_all_kids = [];
% good_subs_kids = [1:13,16,17,19,20,21:25,27];
good_subs_kids = [1:27];

for sub_i = good_subs_kids
    sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0'
    TFS_results_kids.ID(height(TFS_results_kids)+1) = {sub};

    
    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';

    datadir = [project_dir,'Data',filesep,'BIDS',filesep];

    path_Tstat = [datadir,'derivatives',filesep,'Tstats',extension_T,filesep,'sub-',sub,filesep];

    %% envelopes
    load([path_Tstat,'D1_peak_env_13_30_Hz.mat'],'mean_Env_index','mean_Env_index_pinkytrls')
    load([path_Tstat,'D4_peak_env_13_30_Hz.mat'],'mean_Env_pinky','mean_Env_pinky_indextrls')
    %     load([path_Tstat,'D1_peak_env_8_25_Hz.mat'],'mean_Env_index','mean_Env_index_pinkytrls')
    %     load([path_Tstat,'D4_peak_env_8_25_Hz.mat'],'mean_Env_pinky','mean_Env_pinky_indextrls')

    mean_Env_index_all_kids = cat(2,mean_Env_index_all_kids,mean_Env_index);
    mean_Env_index_pinkytrls_all_kids = cat(2,mean_Env_index_pinkytrls_all_kids,mean_Env_index_pinkytrls);
    mean_Env_pinky_all_kids = cat(2,mean_Env_pinky_all_kids,mean_Env_pinky);
    mean_Env_pinky_indextrls_all_kids = cat(2,mean_Env_pinky_indextrls_all_kids,mean_Env_pinky_indextrls);
end

grp_mean_Env_index_all_kids = mean(mean_Env_index_all_kids,2);
grp_mean_Env_index_pinkytrls_all_kids = mean(mean_Env_index_pinkytrls_all_kids,2);
grp_mean_Env_pinky_all_kids = mean(mean_Env_pinky_all_kids,2);
grp_mean_Env_pinky_indextrls_all_kids = mean(mean_Env_pinky_indextrls_all_kids,2);

grp_std_Env_index_all_kids = std(mean_Env_index_all_kids,[],2);
grp_std_Env_index_pinkytrls_all_kids = std(mean_Env_index_pinkytrls_all_kids,[],2);
grp_std_Env_pinky_all_kids = std(mean_Env_pinky_all_kids,[],2);
grp_std_Env_pinky_indextrls_all_kids = std(mean_Env_pinky_indextrls_all_kids,[],2);

diff_window = [1,1.5];
stim_window = [0.3,0.8];

kids_index_post = mean(mean_Env_index_all_kids(diff_window(1)*fs:diff_window(2)*fs,:),1);
adults_index_post = mean(mean_Env_index_all(diff_window(1)*fs:diff_window(2)*fs,:),1);
p_diff_D2_post = ranksum(kids_index_post,adults_index_post)

kids_pinky_post = mean(mean_Env_pinky_all_kids(diff_window*fs:diff_window*fs,:),1);
adults_pinky_post = mean(mean_Env_pinky_all(diff_window*fs:diff_window*fs,:),1);
p_diff_D5_post = ranksum(kids_pinky_post,adults_pinky_post)

% on v post stim modulation
kids_index_on = mean(mean_Env_index_all_kids(stim_window(1)*fs:stim_window(2)*fs,:),1);
adults_index_on = mean(mean_Env_index_all(stim_window(1)*fs:stim_window(2)*fs,:),1);

kids_pinky_on = mean(mean_Env_pinky_all_kids(stim_window*fs:stim_window*fs,:),1);
adults_pinky_on = mean(mean_Env_pinky_all(stim_window*fs:stim_window*fs,:),1);

D2_modulation_kids = kids_index_post - kids_index_on;
D5_modulation_kids = kids_pinky_post - kids_pinky_on;
D2_modulation_adults = adults_index_post - adults_index_on;
D5_modulation_adults = adults_pinky_post - adults_pinky_on;
p_diff_D2_mod = ranksum(D2_modulation_kids,D2_modulation_adults);
p_diff_D5_mod = ranksum(D5_modulation_kids,D5_modulation_adults);

%%
addpath ./Violinplot-Matlab-master/
ylims = [-0.2,0.6];
figure('Color','w','Position',[680 470 630 508])
subplot(1,2,1)
index.Adults = D2_modulation_adults;
index.Children = D2_modulation_kids;
vs = violinplot(index);
ylabel('Stimulus vs Post-stimulus modulation');
title('D2')
box off
% ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
ax.FontSize=FtSz;

if p_diff_D2_mod < 0.05 && p_diff_D2_mod>= 0.01; text(1.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end
if p_diff_D2_mod < 0.01; text(1.5,0.8.*ylims(2),'**','FontSize',40,...
        'HorizontalAlignment', 'center');end

subplot(1,2,2)
pinky.Adults = D5_modulation_adults;
pinky.Children = D5_modulation_kids;
vs = violinplot(pinky);
ylabel('Stimulus vs Post-stimulus modulation');
title('D5')
box off
% ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
ax.FontSize=FtSz;

if p_diff_D5_mod < 0.05 && p_diff_D5_mod>= 0.01; text(1.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end
if p_diff_D5_mod < 0.01; text(1.5,0.8.*ylims(2),'**','FontSize',40,...
        'HorizontalAlignment', 'center');end


%%   modulations
ylims = [-0.2,0.6];
figure('Color','w','Units','centimeters')
set(gcf,'Position',[10,10,12,17.5])
data.D2_Adults = D2_modulation_adults;
data.D2_Children = D2_modulation_kids;
data.D5_Adults = D5_modulation_adults;
data.D5_Children = D5_modulation_kids;

vs = violinplot(data);
vs(3).ViolinColor = vs(1).ViolinColor
vs(4).ViolinColor = vs(2).ViolinColor
ylabel('Stim. vs Post-stim. modulation');
box off
% ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
ax.FontSize=25;
axis fill
if p_diff_D2_mod < 0.05 && p_diff_D2_mod>= 0.01; text(1.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end
if p_diff_D2_mod < 0.01; text(3.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end
if p_diff_D5_mod < 0.05 && p_diff_D5_mod>= 0.01; text(1.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end
if p_diff_D5_mod < 0.01; text(3.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end
% xticklabels({'D2_{Adults}','D2_{Children}','D5_{Adults}','D5_{Children}'})
xticklabels({'Adults (D2)','Children (D2)','Adults (D5)','Children (D5)'})
xticklabels({'D2','D2','D5','D5'})
ylim([-.3,1])
% ylim(ylims)
xlim(xlim + [-.1,.1])



%%
figure
FtSz=15;
set(gcf,'Color','w')
set(gcf,'Position',[681 559 1120 420])

fs = 1200;
time = linspace(0,size(grp_mean_Env_index_all_kids,1)./fs,size(grp_mean_Env_index_all_kids,1));
hold on

ylims=[-0.4000 0.5484];
actwin = [0.3,0.8];
control_period = [conwin(1),ylims(1);conwin(1),ylims(2);conwin(2),ylims(2);...
    conwin(2),ylims(1);conwin(1),ylims(1)];
active_period = [actwin(1),ylims(1);actwin(1),ylims(2);actwin(2),ylims(2);...
    actwin(2),ylims(1);actwin(1),ylims(1)];

ph(1) = fill(control_period(:,1),control_period(:,2),[.5,.5,.5],'FaceAlpha',0.4,'DisplayName','Rest');
ph(2) = fill(active_period(:,1),active_period(:,2),'g','FaceAlpha',0.4,'DisplayName','Active');
set(ph,'EdgeColor','none','HandleVisibility','off')

plot(time,grp_mean_Env_index_all_kids,'r','LineWidth',2,...
    'Displayname','D2 desync peak');
ciplot(grp_mean_Env_index_all_kids-grp_std_Env_index_all_kids,...
    grp_mean_Env_index_all_kids+grp_std_Env_index_all_kids,time,'r')

plot(time,grp_mean_Env_pinky_all_kids,'b','LineWidth',2,...
    'Displayname','D5 desync peak');
ciplot(grp_mean_Env_pinky_all_kids-grp_std_Env_pinky_all_kids,...
    grp_mean_Env_pinky_all_kids+grp_std_Env_pinky_all_kids,time,'b')
lh = legend;lh.String{2} = 'std. dev.';lh.String{4} = 'std. dev.';
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;chldrn(3).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Fractional Change')

%%
fs = 1200;
control_window = round(conwin.*fs);control_inds = control_window(1):control_window(2);

f_D2_all_kids = figure;f_D2_all_kids.Color = 'w';set(f_D2_all_kids,"Units","normalized","Position",[0,0,1,1])
TL_D2_kids = tiledlayout('flow','TileSpacing','Compact');
f_D5_all_kids = figure;f_D5_all_kids.Color = 'w';set(f_D5_all_kids,"Units","normalized","Position",[0,0,1,1])
TL_D5_kids = tiledlayout('flow','TileSpacing','Compact');
TFS_all_D2_kids = [];
TFS_all_D5_kids = [];
for sub_i = good_subs_kids
    sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0'
    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';

    datadir = [project_dir,'Data',filesep,'BIDS',filesep];

    path_Tstat = [datadir,'derivatives',filesep,'Tstats',extension_T,filesep,'sub-',sub,filesep];
    % load timecourses
    load([path_Tstat,'D1_peak_VE_1-150_Hz.mat'],'VE_index','VE_index_pinkytrls')
    load([path_Tstat,'D4_peak_VE_1-150_Hz.mat'],'VE_pinky','VE_pinky_indextrls')

    [TFS,fre] = VE_TFS(VE_index,control_inds,time,fs);
    TFS_all_D2_kids = cat(3,TFS_all_D2_kids,TFS);

    figure(f_D2_all_kids)
    nexttile
    pcolor(time,fre,TFS_all_D2_kids(:,:,end));shading interp
    xlabel('Time (s)');ylabel('Frequency (Hz)')
    colorbar;cbar_lim = [0.5];
    caxis([-cbar_lim cbar_lim])
    axis fill
    title([sub])


    [TFS,~] = VE_TFS(VE_pinky,control_inds,time,fs);
    TFS_all_D5_kids = cat(3,TFS_all_D5_kids,TFS);

    figure(f_D5_all_kids)
    nexttile
    pcolor(time,fre,TFS_all_D5_kids(:,:,end));shading interp
    xlabel('Time (s)');ylabel('Frequency (Hz)')
    colorbar;
    caxis([-cbar_lim cbar_lim])
    axis fill
    title([sub])
end
figure(f_D2_all_kids)
sgtitle('D2')
figure(f_D5_all_kids)
sgtitle('D5')
set(TL_D2_kids.Children(2:2:end),'YLim',[5,50])
set(TL_D5_kids.Children(2:2:end),'YLim',[5,50])

%%
FntS=15;
figure
set(gcf,'Color','w','Position',[681 510 900 400])
cbar_lim = [0.2];
subplot(1,2,1)
pcolor(time,fre,mean(TFS_all_D2_kids,3));shading interp
xlabel('Time (s)');
ylabel('Frequency (Hz)')
cb = colorbar;
caxis([-cbar_lim cbar_lim])
% cb.Label.String = 'Fractional change';cb.Label.FontSize = FntS;
axis fill
ax=gca;
ax.FontSize=FntS;ax.YLabel.FontSize = FntS;
ax.XLabel.FontSize = FntS;
ylim([5,50])
title('D1')

subplot(1,2,2)
pcolor(time,fre,mean(TFS_all_D5_kids,3));shading interp
xlabel('Time (s)');%ylabel('Frequency (Hz)')
cb = colorbar;caxis([-cbar_lim cbar_lim])
cb.Label.String = 'Fractional change';cb.Label.FontSize = FntS;
axis fill
set(gcf,'Name','D4 TFS')
ax=gca;
ax.FontSize=FntS;ax.YLabel.FontSize = FntS;
ax.XLabel.FontSize = FntS;
ylim([5,50])
title('D5')
sgtitle('Kids')

%% Mean TFS both fingers
FntS=17;
figure
set(gcf,'Color','w','Units','centimeters','Position',[10,10,11.5,14.5])
cbar_lim = [0.2];
subplot(2,1,1)
mean_kids_TFS = (mean(TFS_all_D2_kids,3));% + mean(TFS_all_D5_kids,3))./2;
pcolor(time,fre,mean_kids_TFS);shading interp
xlabel('Time (s)');
ylabel('Frequency (Hz)')
cb = colorbar;cb.FontSize=12;cb.Label.Position(1)=2.5;
caxis([-cbar_lim cbar_lim])
cb.Label.String = 'Fractional change';
axis fill
ax=gca;
ax.FontSize=FntS;ax.YLabel.FontSize = FntS;
ax.XLabel.FontSize = FntS;
ylim([5,50])

subplot(2,1,2)
mean_adults_TFS = (mean(TFS_all_D2,3));% + mean(TFS_all_D5,3))./2;
pcolor(time,fre,mean_adults_TFS);shading interp
xlabel('Time (s)');
ylabel('Frequency (Hz)')
cb = colorbar;caxis([-cbar_lim cbar_lim]);cb.FontSize=15;cb.Label.Position(1)=2.5;
cb.Label.String = 'Fractional change';
axis fill
set(gcf,'Name','Both group average TFS')
ax=gca;
ax.FontSize=FntS;ax.YLabel.FontSize = FntS;
ax.XLabel.FontSize = FntS;
ylim([5,50])

%% compare broth envs

%
addpath ./Violinplot-Matlab-master/
ylims = [-0.2,0.6];
figure('Color','w','Position',[680 470 630 508])
subplot(1,2,1)
index.Adults = adults_index_post;
index.Children = kids_index_post;
vs = violinplot(index);
ylabel('Post-stimulus % ch. from BL');
title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
ax.FontSize=FtSz;

if p_diff_D2_post < 0.01; text(1.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end

subplot(1,2,2)
pinky.Adults = adults_pinky_post;
pinky.Children = kids_pinky_post;
vs = violinplot(pinky);
ylabel('Post-stimulus % ch. from BL');
title('D5')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
ax.FontSize=FtSz;

if p_diff_D5_post < 0.01; text(1.5,0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end

adult_col = [0 0.4470 0.7410];
kid_col = [0.8500 0.3250 0.0980];

%% %%%% Envelopes
figure
FtSz=25;
set(gcf,'Color','w','Units','centimeters','Position',[0,0,21.4,17.5])


fs = 1200;
time = linspace(0,size(grp_mean_Env_index_all,1)./fs,size(grp_mean_Env_index_all,1));
% subplot(2,1,1)
hold on

ylims=[-0.45 0.45];
actwin = [0.3,0.8];
control_period = [conwin(1),ylims(1);conwin(1),ylims(2);conwin(2),ylims(2);...
    conwin(2),ylims(1);conwin(1),ylims(1)];
active_period = [actwin(1),ylims(1);actwin(1),ylims(2);actwin(2),ylims(2);...
    actwin(2),ylims(1);actwin(1),ylims(1)];
% ph(1) = fill(control_period(:,1),control_period(:,2),.5.*[1,1,1],'FaceAlpha',0.3,'DisplayName','Rest');
% ph(2) = fill(active_period(:,1),active_period(:,2),'g','FaceAlpha',0.2,'DisplayName','Active');
% set(ph(1:2),'EdgeColor','none','HandleVisibility','off')
diff_period = [diff_window(1),ylims(1);diff_window(1),ylims(2);diff_window(2),ylims(2);...
    diff_window(2),ylims(1);diff_window(1),ylims(1)];
fill(diff_period(:,1),diff_period(:,2),[1,1,1].*0.8,'FaceAlpha',0.4,'DisplayName','Active','EdgeColor','none','HandleVisibility','off');
fill(active_period(:,1),active_period(:,2),'g','FaceAlpha',0.4,'DisplayName','Active','EdgeColor','none','HandleVisibility','off');
plot(time,grp_mean_Env_index_all,'Color',adult_col,'LineWidth',2,...
    'Displayname','D2 ERD | Adults');
ciplot(grp_mean_Env_index_all-grp_std_Env_index_all,...
    grp_mean_Env_index_all+grp_std_Env_index_all,time,adult_col)

plot(time,grp_mean_Env_index_all_kids,'Color',kid_col,'LineWidth',2,...
    'Displayname','D2 ERD | Children');
ciplot(grp_mean_Env_index_all_kids-grp_std_Env_index_all_kids,...
    grp_mean_Env_index_all_kids+grp_std_Env_index_all_kids,time,kid_col)


lh = legend;lh.String{2} = 'std. dev.';lh.String{4} = 'std. dev.';
lh.FontSize = 15;
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;chldrn(3).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Fractional Change')
% if p_diff_D2_post < 0.01; text(mean(diff_window),0.8.*ylims(2),'*','FontSize',40,...
%         'HorizontalAlignment', 'center');end
xlim([0,3.5])
lh.Location = 'southeast';
%%

%%%%
subplot(2,1,2)
hold on
% ph(3) = fill(control_period(:,1),control_period(:,2),[.5,.5,.5],'FaceAlpha',0.4,'DisplayName','Rest');
% ph(4) = fill(active_period(:,1),active_period(:,2),'g','FaceAlpha',0.4,'DisplayName','Active');
% set(ph(3:4),'EdgeColor','none','HandleVisibility','off')
fill(diff_period(:,1),diff_period(:,2),[1,1,1].*0.8,'FaceAlpha',0.4,'DisplayName','Active','EdgeColor','none','HandleVisibility','off');
fill(active_period(:,1),active_period(:,2),'g','FaceAlpha',0.4,'DisplayName','Active','EdgeColor','none','HandleVisibility','off');

plot(time,grp_mean_Env_pinky_all,'Color',adult_col,'LineWidth',2,...
    'Displayname','D5 ERD | Adults');
ciplot(grp_mean_Env_pinky_all-grp_std_Env_pinky_all,...
    grp_mean_Env_pinky_all+grp_std_Env_pinky_all,time,adult_col)
plot(time,grp_mean_Env_pinky_all_kids,'Color',kid_col,'LineWidth',2,...
    'Displayname','D5 ERD | Children');
ciplot(grp_mean_Env_pinky_all_kids-grp_std_Env_pinky_all_kids,...
    grp_mean_Env_pinky_all_kids+grp_std_Env_pinky_all_kids,time,kid_col)
lh = legend;lh.String{2} = 'std. dev.';lh.String{4} = 'std. dev.';
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;chldrn(3).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Fractional Change')
if p_diff_D5_post < 0.01;text(mean(diff_window),0.8.*ylims(2),'*','FontSize',40,...
        'HorizontalAlignment', 'center');end

TFS_results_adults.TFS_D2 = squeeze(mat2cell(TFS_all_D2,size(TFS_all_D2,1),size(TFS_all_D2,2),ones(1,size(TFS_all_D2,3))));
TFS_results_kids.TFS_D2 = squeeze(mat2cell(TFS_all_D2_kids,size(TFS_all_D2_kids,1),size(TFS_all_D2_kids,2),ones(1,size(TFS_all_D2_kids,3))));
TFS_results_adults.TFS_D5 = squeeze(mat2cell(TFS_all_D5,size(TFS_all_D5,1),size(TFS_all_D5,2),ones(1,size(TFS_all_D5,3))));
TFS_results_kids.TFS_D5 = squeeze(mat2cell(TFS_all_D5_kids,size(TFS_all_D5_kids,1),size(TFS_all_D5_kids,2),ones(1,size(TFS_all_D5_kids,3))));

TFS_results_adults.beta_env_D2 = mean_Env_index_all';
TFS_results_kids.beta_env_D2 = mean_Env_index_all_kids';
TFS_results_adults.beta_env_D5 = mean_Env_pinky_all';
TFS_results_kids.beta_env_D5 = mean_Env_pinky_all_kids';

TFS_results_adults.beta_modulation_D2 = D2_modulation_adults';
TFS_results_adults.beta_modulation_D5 = D5_modulation_adults';
TFS_results_kids.beta_modulation_D2 = D2_modulation_kids';
TFS_results_kids.beta_modulation_D5 = D5_modulation_kids';

if badtrls
    save('TFS_RESULTS_BETA_badtrl.mat','TFS_results_adults','TFS_results_kids')
else
    save('TFS_RESULTS_BETA.mat','TFS_results_adults','TFS_results_kids')
end

%% % -----------------------------------------------------------------------
%%% subfunctions
%%% -----------------------------------------------------------------------
function [TFS,fre] = VE_TFS(VE_chopped,conwin,trial_time,fs)
cbar_lim = [0.3]; % colour bar limit (relative change)
% highpass = [1 2 4 6 8 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110];
% lowpass = [4 6 8 10 13 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120];
% fre = highpass + ((lowpass - highpass)./2);
fre = 3:2:50;
highpass = fre - 1.5;
lowpass = fre + 1.5;
% Control window
% fx = figure;
% fx.Name = 'VE TFS';
% Filter data within bands and calculate envelope
VE_unchopped = reshape(VE_chopped,1,size(VE_chopped,1)*size(VE_chopped,2));
VE_fb = zeros(length(VE_unchopped),length(fre));
for fb = 1:length(highpass)
    filt_VE = nut_filter3(VE_unchopped','butter','bp',4,highpass(fb),lowpass(fb),fs,1)';
    VE_fb(:,fb) = abs(hilbert(filt_VE));
end
VE_mean = zeros(size(VE_chopped,1),length(fre));
for fb = 1:length(highpass)
    % Chop data
    VE_filt = reshape(VE_fb(:,fb),size(VE_chopped,1),size(VE_chopped,2));
    % Average across trials
    VE_mean(:,fb) = mean(VE_filt,2);
end
meanrest = mean(VE_mean(conwin,:),1);
meanrestmat = repmat(meanrest,size(VE_mean,1),1);
TFS = (VE_mean'-meanrestmat')./meanrestmat';
% figure(fx)
% pcolor(trial_time,fre,TFS);shading interp
% xlabel('Time (s)');ylabel('Frequency (Hz)')
% colorbar;caxis([-cbar_lim cbar_lim])
% axis fill
end
