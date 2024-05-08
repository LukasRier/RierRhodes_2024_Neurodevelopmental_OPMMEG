clearvars
addpath ./Violinplot-Matlab-master/
addpath .\BrainPlots
badtrl = 0
if badtrl
    results_dir = '.\Figs_badtrl\';
    load TFS_RESULTS_BETA_badtrl.mat
    load AEC_RESULTS_badtrl.mat
    load HMM_RESULTS_badtrl.mat
    load('Evoked_traces_badtrl.mat',"grp_mean_evoked*index","grp_mean_evoked*pinky","grp_std_evoked*index","grp_std_evoked*pinky")
    load EVOKED_RESULTS_badtrl.mat
else
    results_dir = '.\Figs\';
    load TFS_RESULTS_BETA.mat
    load AEC_RESULTS.mat
    load HMM_RESULTS.mat
    load('Evoked_traces.mat',"grp_mean_evoked*index","grp_mean_evoked*pinky","grp_std_evoked*index","grp_std_evoked*pinky")
    load EVOKED_RESULTS.mat

end

if not(all(startsWith(hmm_results_kids.ID,AEC_results_kids.ID)) & ...
all(startsWith(hmm_results_kids.ID,TFS_results_kids.ID)) & ...
all(startsWith(hmm_results_kids.ID,EVOKED_results_kids.ID)) & ...
all(startsWith(hmm_results_adults.ID,AEC_results_adults.ID)) & ...
all(startsWith(hmm_results_adults.ID,TFS_results_adults.ID)) & ...
all(startsWith(hmm_results_adults.ID,EVOKED_results_adults.ID)))
    error('Results not consistent!')
end


%% data quality
load('data_quality_adults.mat')
data_quality_adults = data_quality; 
load('data_quality_children.mat');
data_quality_children = data_quality;
f_DQ = fopen('data_quality.txt','w');
fprintf(f_DQ,"\n\n#####################################" + ...
    "\nData quality metrics\n\n");
fprintf(f_DQ,'Good channels (adults) = %1.2f +- %1.2f\n',mean(data_quality_adults.good_chans),...
std(data_quality_adults.good_chans));
fprintf(f_DQ,'Good channels (children) = %1.2f +- %1.2f\n',mean(data_quality_children.good_chans),...
std(data_quality_children.good_chans));
fprintf(f_DQ,'Good channels (all) = %1.2f +- %1.2f\n\n',mean([data_quality_adults.good_chans;data_quality_children.good_chans]),...
std([data_quality_adults.good_chans;data_quality_children.good_chans]));

fprintf(f_DQ,'Bad channels (adults) = %1.2f +- %1.2f\n',mean(data_quality_adults.N_bad_chans),...
std(data_quality_adults.N_bad_chans));
fprintf(f_DQ,'Bad channels (children) = %1.2f +- %1.2f\n\n',mean(data_quality_children.N_bad_chans),...
std(data_quality_children.N_bad_chans));

fprintf(f_DQ,'Bad trials D2 (adults) = %1.2f +- %1.2f of 42\n',42-mean(data_quality_adults.good_trls_D2),...
std(data_quality_adults.good_trls_D2));
fprintf(f_DQ,'Bad trials D2 (adults) percentage = %1.2f +- %1.2f%%\n\n',100-mean(data_quality_adults.good_trls_D2)./(42/100),...
std(data_quality_adults.good_trls_D2)./(42/100));

fprintf(f_DQ,'Bad trials D2 (children) = %1.2f +- %1.2f of 42\n',42-mean(data_quality_children.good_trls_D2),...
std(data_quality_children.good_trls_D2));
fprintf(f_DQ,'Bad trials D2 (children) percentage = %1.2f +- %1.2f%%\n\n',100-mean(data_quality_children.good_trls_D2)./(42/100),...
std(data_quality_children.good_trls_D2)./(42/100));

fprintf(f_DQ,'Bad trials D5 (adults) = %1.2f +- %1.2f of 42\n',42-mean(data_quality_adults.good_trls_D5),...
std(data_quality_adults.good_trls_D5));
fprintf(f_DQ,'Bad trials D5 (adults) percentage = %1.2f +- %1.2f%%\n\n',100-mean(data_quality_adults.good_trls_D5)./(42/100),...
std(data_quality_adults.good_trls_D5)./(42/100));

fprintf(f_DQ,'Bad trials D5 (children) = %1.2f +- %1.2f of 42\n',42-mean(data_quality_children.good_trls_D5),...
std(data_quality_children.good_trls_D5));
fprintf(f_DQ,'Bad trials D5 (children) percentage = %1.2f +- %1.2f%%\n\n',100-mean(data_quality_children.good_trls_D5)./(42/100),...
std(data_quality_children.good_trls_D5)./(42/100));
fclose(f_DQ);


%%
kids_ages = hmm_results_kids.age;
[~,age_rank] = sort(kids_ages);

youngest = age_rank(1:floor(length(age_rank)/3));
oldest = age_rank(end - floor(length(age_rank)/3)+1 : end);
middle = age_rank(floor(length(age_rank)/3)+1:end - floor(length(age_rank)/3));
adult_ages = hmm_results_adults.age;

[~,age_rank_ad] = sort(adult_ages);

youngest_ad = age_rank_ad(1:floor(length(age_rank_ad)/3));
oldest_ad = age_rank_ad(end - floor(length(age_rank_ad)/3)+1 : end);
middle_ad = age_rank_ad(floor(length(age_rank_ad)/3)+1:end - floor(length(age_rank_ad)/3));

%% TFS
youngest_TFS_D2 = reshape([TFS_results_kids.TFS_D2{youngest}],[size(TFS_results_kids.TFS_D2{1}),length(youngest)]);
youngest_TFS_D5 = reshape([TFS_results_kids.TFS_D5{youngest}],[size(TFS_results_kids.TFS_D5{1}),length(youngest)]);
middle_TFS_D2 = reshape([TFS_results_kids.TFS_D2{middle}],[size(TFS_results_kids.TFS_D2{1}),length(middle)]);
middle_TFS_D5 = reshape([TFS_results_kids.TFS_D5{middle}],[size(TFS_results_kids.TFS_D5{1}),length(middle)]);
oldest_TFS_D2 = reshape([TFS_results_kids.TFS_D2{oldest}],[size(TFS_results_kids.TFS_D2{1}),length(oldest)]);
oldest_TFS_D5 = reshape([TFS_results_kids.TFS_D5{oldest}],[size(TFS_results_kids.TFS_D5{1}),length(oldest)]);

youngest_TFS_D2_ad = reshape([TFS_results_adults.TFS_D2{youngest_ad}],[size(TFS_results_adults.TFS_D2{1}),length(youngest_ad)]);
youngest_TFS_D5_ad = reshape([TFS_results_adults.TFS_D5{youngest_ad}],[size(TFS_results_adults.TFS_D5{1}),length(youngest_ad)]);
middle_TFS_D2_ad = reshape([TFS_results_adults.TFS_D2{middle_ad}],[size(TFS_results_adults.TFS_D2{1}),length(middle_ad)]);
middle_TFS_D5_ad = reshape([TFS_results_adults.TFS_D5{middle_ad}],[size(TFS_results_adults.TFS_D5{1}),length(middle_ad)]);
oldest_TFS_D2_ad = reshape([TFS_results_adults.TFS_D2{oldest_ad}],[size(TFS_results_adults.TFS_D2{1}),length(oldest_ad)]);
oldest_TFS_D5_ad = reshape([TFS_results_adults.TFS_D5{oldest_ad}],[size(TFS_results_adults.TFS_D5{1}),length(oldest_ad)]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PAPER Fig 3

fh = figure;
set(fh,'Units','centimeters','Color','w','Renderer','painters');
fwidth = 8.8239;
fheight = 4;
fh.Position([3,4]) = [fwidth,fheight];
cbar_lim = 0.3;

plot_TFS(mean(youngest_TFS_D2,3),cbar_lim,grp_mean_evoked_youngest_index,grp_std_evoked_youngest_index)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
%
print(gcf,sprintf('%sFig3/Youngest_D2_TFS.png',results_dir),'-dpng','-r900');
clf
%% waitforbuttonpress

plot_TFS(mean(middle_TFS_D2,3),cbar_lim,grp_mean_evoked_middle_index,grp_std_evoked_middle_index)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
print(gcf,sprintf('%sFig3/Middle_D2_TFS.png',results_dir),'-dpng','-r900');
clf
% waitforbuttonpress

plot_TFS(mean(oldest_TFS_D2,3),cbar_lim,grp_mean_evoked_oldest_index,grp_std_evoked_oldest_index)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
print(gcf,sprintf('%sFig3/Oldest_D2_TFS.png',results_dir),'-dpng','-r900');

%% waitforbuttonpress
clf
plot_TFS(mean(cat(3,youngest_TFS_D2_ad,middle_TFS_D2_ad,oldest_TFS_D2_ad),3),cbar_lim,grp_mean_evoked_index,grp_std_evoked_index)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
print(gcf,sprintf('%sFig3/AllAdults_D2_TFS.png',results_dir),'-dpng','-r900');

%% D5
fh = figure;
set(fh,'Units','centimeters','Color','w','Renderer','painters');
fh.Position([3,4]) = [fwidth,fheight];
cbar_lim = 0.3;

plot_TFS(mean(youngest_TFS_D5,3),cbar_lim,grp_mean_evoked_youngest_pinky,grp_std_evoked_youngest_pinky)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
print(gcf,sprintf('%sFig3/Youngest_D5_TFS.png',results_dir),'-dpng','-r900');
clf
% waitforbuttonpress

plot_TFS(mean(middle_TFS_D5,3),cbar_lim,grp_mean_evoked_middle_pinky,grp_std_evoked_middle_pinky)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
print(gcf,sprintf('%sFig3/Middle_D5_TFS.png',results_dir),'-dpng','-r900');
clf
% waitforbuttonpress

plot_TFS(mean(oldest_TFS_D5,3),cbar_lim,grp_mean_evoked_oldest_pinky,grp_std_evoked_oldest_pinky)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
print(gcf,sprintf('%sFig3/Oldest_D5_TFS.png',results_dir),'-dpng','-r900');
clf
% waitforbuttonpress

plot_TFS(mean(cat(3,youngest_TFS_D5_ad,middle_TFS_D5_ad,oldest_TFS_D5_ad),3),cbar_lim,grp_mean_evoked_pinky,grp_std_evoked_pinky)
delete(fh.Children(1).Label)
fh.Children(1).TickLabels = [];
fh.Children(1).Position(2) = 0.05;
fh.Children(1).Position(1) = 0.1;
fh.Children(2).Position(4) = 0.55;
fh.Children(2).Position(2) = 0.32;
fh.Children(2).Position(1) = 0.1;
set(fh.Children,'FontSize',9,'FontName','Arial')
print(gcf,sprintf('%sFig3/AllAdults_D5_TFS.png',results_dir),'-dpng','-r900');
%%
%% % beta envelope modulation w age
f_mod_v_age = figure('Name','Beta Mod v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 6;
fheight = 6;
if badtrl
fwidth = 8;
fheight = 6;
end
f_mod_v_age.Position([3,4]) = [fwidth,fheight];

scatter(kids_ages,TFS_results_kids.beta_modulation_D2,'k^')
hold on
scatter(adult_ages,TFS_results_adults.beta_modulation_D2,'ko')
ylabel('\beta-modulation (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])
include_mm = 0
if include_mm
    xlim([0,68])
    MMmod = 37.1013710565052/100
    mmh=scatter(66,MMmod,'bo');
    mmh.HandleVisibility = 'off';
    mmh.MarkerFaceColor = 'b';
end
ylim([-0.1,0.7])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [TFS_results_kids.beta_modulation_D2;TFS_results_adults.beta_modulation_D2];
b = X\y;
[R_mod_D2,p_mod_D2] = corr(x,y);
refl = refline(b(2),b(1));

set(f_mod_v_age.Children,'FontName','Arial','FontSize',9)
print(gcf,sprintf('%sFig3/Beta_mod_D2.png',results_dir),'-dpng','-r900');

legend({'Children','Adults',sprintf('R^2 = %1.2f\np = %1.5f',R_mod_D2^2,p_mod_D2)}, ...
        'box','on','Location','eastoutside','NumColumns',1)
print(gcf,sprintf('%sFig3/Beta_mod_D2_w_legend.png',results_dir),'-dpng','-r900');

% D5 %%%%%%%%%%%%%%%%%%
f_mod_v_age = figure('Name','Beta Mod v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');

f_mod_v_age.Position([3,4]) = [fwidth,fheight];

scatter(kids_ages,TFS_results_kids.beta_modulation_D5,'k^')
hold on
scatter(adult_ages,TFS_results_adults.beta_modulation_D5,'ko')
ylabel('\beta-modulation (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

ylim([-0.1,0.7])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [TFS_results_kids.beta_modulation_D5;TFS_results_adults.beta_modulation_D5];
b = X\y;
[R_mod_D5,p_mod_D5] = corr(x,y);
refl = refline(b(2),b(1));

set(f_mod_v_age.Children,'FontName','Arial','FontSize',9)
print(gcf,sprintf('%sFig3/Beta_mod_D5.png',results_dir),'-dpng','-r900');

legend({'Children','Adults',sprintf('R^2 = %1.2f\np = %1.5f',R_mod_D5^2,p_mod_D5)}, ...
        'box','on','Location','eastoutside','NumColumns',1)
print(gcf,sprintf('%sFig3/Beta_mod_D5_w_legend.png',results_dir),'-dpng','-r900');

%% evoked scatter plots
% load('EVOKED_RESULTS.mat','EVOKED_results_adults','EVOKED_results_kids','t_ind_p50')


f_p50_v_age = figure('Name','P50 v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');

f_p50_v_age.Position([3,4]) = [fwidth,fheight];

scatter(kids_ages,EVOKED_results_kids.evoked_D2(:,t_ind_p50),'k^')
hold on
scatter(adult_ages,EVOKED_results_adults.evoked_D2(:,t_ind_p50),'ko')
ylabel('P50 Amplitude (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [EVOKED_results_kids.evoked_D2(:,t_ind_p50);EVOKED_results_adults.evoked_D2(:,t_ind_p50)];
b = X\y;
[R_P50_D2,p_P50_D2] = corr(x,y);
refl = refline(b(2),b(1));

set(f_p50_v_age.Children,'FontName','Arial','FontSize',9)
print(gcf,sprintf('%sFig3/P50_v_age_D2.png',results_dir),'-dpng','-r900');

legend({'Children','Adults',sprintf('R^2 = %1.2f\np = %1.5f',R_P50_D2^2,p_P50_D2)}, ...
        'box','on','Location','eastoutside','NumColumns',1)
print(gcf,sprintf('%sFig3/P50_v_age_D2_w_legend.png',results_dir),'-dpng','-r900');


f_p50_v_age = figure('Name','P50 v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');

%D5
f_p50_v_age.Position([3,4]) = [fwidth,fheight];

scatter(kids_ages,EVOKED_results_kids.evoked_D5(:,t_ind_p50),'k^')
hold on
scatter(adult_ages,EVOKED_results_adults.evoked_D5(:,t_ind_p50),'ko')
ylabel('P50 Amplitude (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [EVOKED_results_kids.evoked_D5(:,t_ind_p50);EVOKED_results_adults.evoked_D5(:,t_ind_p50)];
b = X\y;
[R_P50_D5,p_P50_D5] = corr(x,y);
refl = refline(b(2),b(1));

set(f_p50_v_age.Children,'FontName','Arial','FontSize',9)
print(gcf,sprintf('%sFig3/P50_v_age_D5.png',results_dir),'-dpng','-r900');

legend({'Children','Adults',sprintf('R^2 = %1.2f\np = %1.5f',R_P50_D5^2,p_P50_D5)}, ...
        'box','on','Location','eastoutside','NumColumns',1)
print(gcf,sprintf('%sFig3/P50_v_age_D5_w_legend.png',results_dir),'-dpng','-r900');

%% age group maps

addpath('R:\DRS-KidsOPM\Paediatric_OPM_Notts\fieldtrip-20220906')
ft_defaults;
ses = '001';
run = 'run-001';
exp_type = '_task-braille';
Tstat_results_adults = table({},...
    'VariableNames',{'ID'});
Tstat_results_kids = table({},...
    'VariableNames',{'ID'});

project_dir =  'R:\DRS-KidsOPM\Paediatric_OPM_Notts\';
datadir = [project_dir,'Data',filesep,'BIDS',filesep];

for subi = 1:height(AEC_results_kids)
    sub = AEC_results_kids.ID{subi};
    Tstat_results_kids.ID{subi} = AEC_results_kids.ID{subi};
    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];

    path_Tstat = [datadir,'derivatives',filesep,'Tstats',filesep,'sub-',sub,filesep];
    files_Tstat = [filename,'_pseudoT_'];

    image = ft_read_mri([path_Tstat,files_Tstat,'index_MNI.nii.gz']);
    Tstat_results_kids.Tstat_D2{subi} = image.anatomy;

    image = ft_read_mri([path_Tstat,files_Tstat,'pinky_MNI.nii.gz']);
    Tstat_results_kids.Tstat_D5{subi} = image.anatomy;
end

project_dir =  'R:\DRS-KidsOPM\Paediatric_OPM_Notts_AdultData\';
datadir = [project_dir,'Data',filesep,'BIDS',filesep];

for subi = 1:height(AEC_results_adults)
    sub = AEC_results_adults.ID{subi};
    Tstat_results_adults.ID{subi} = AEC_results_adults.ID{subi};
    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];

    path_Tstat = [datadir,'derivatives',filesep,'Tstats',filesep,'sub-',sub,filesep];
    files_Tstat = [filename,'_pseudoT_'];

    image = ft_read_mri([path_Tstat,files_Tstat,'index_MNI.nii.gz']);
    Tstat_results_adults.D2_Tstat{subi} = image.anatomy;

    image = ft_read_mri([path_Tstat,files_Tstat,'pinky_MNI.nii.gz']);
    Tstat_results_adults.D5_Tstat{subi} = image.anatomy;
end

kids_Tstat_D2 = [];
kids_Tstat_D5 = [];
adults_Tstat_D2 = [];
adults_Tstat_D5 = [];

for subi = 1:height(AEC_results_kids)
    kids_Tstat_D2 = cat(4,kids_Tstat_D2,Tstat_results_kids.Tstat_D2{subi});
    kids_Tstat_D5 = cat(4,kids_Tstat_D5,Tstat_results_kids.Tstat_D5{subi});
end

for subi = 1:height(AEC_results_adults)
    adults_Tstat_D2 = cat(4,adults_Tstat_D2,Tstat_results_adults.D2_Tstat{subi});
    adults_Tstat_D5 = cat(4,adults_Tstat_D5,Tstat_results_adults.D5_Tstat{subi});
end


youngest_Tstat_D2 = kids_Tstat_D2(:,:,:,youngest);
youngest_Tstat_D5 = kids_Tstat_D5(:,:,:,youngest);
middle_Tstat_D2 = kids_Tstat_D2(:,:,:,middle);
middle_Tstat_D5 = kids_Tstat_D5(:,:,:,middle);
oldest_Tstat_D2 = kids_Tstat_D2(:,:,:,oldest);
oldest_Tstat_D5 = kids_Tstat_D5(:,:,:,oldest);

youngest_Tstat_D2_ad = adults_Tstat_D2(:,:,:,youngest_ad);
youngest_Tstat_D5_ad = adults_Tstat_D5(:,:,:,youngest_ad);
middle_Tstat_D2_ad = adults_Tstat_D2(:,:,:,middle_ad);
middle_Tstat_D5_ad = adults_Tstat_D5(:,:,:,middle_ad);
oldest_Tstat_D2_ad = adults_Tstat_D2(:,:,:,oldest_ad);
oldest_Tstat_D5_ad = adults_Tstat_D5(:,:,:,oldest_ad);
all_adults_D2 = cat(4,youngest_Tstat_D2_ad,middle_Tstat_D2_ad,oldest_Tstat_D2_ad);
all_adults_D5 = cat(4,youngest_Tstat_D5_ad,middle_Tstat_D5_ad,oldest_Tstat_D5_ad);

image.anatomy = mean(all_adults_D2,4);
ft_write_mri(sprintf('%sFig3/AllAdults_D2_Tstat.nii',results_dir),image,...
    'dataformat','nifti');
image.anatomy = mean(all_adults_D5,4);
ft_write_mri(sprintf('%sFig3/AllAdults_D5_Tstat.nii',results_dir),image,...
    'dataformat','nifti');

image.anatomy = mean(youngest_Tstat_D2,4);
ft_write_mri(sprintf('%sFig3/Youngest_D2_Tstat.nii',results_dir),image,...
    'dataformat','nifti');
image.anatomy = mean(youngest_Tstat_D5,4);
ft_write_mri(sprintf('%sFig3/Youngest_D5_Tstat.nii',results_dir),image,...
    'dataformat','nifti');

image.anatomy = mean(middle_Tstat_D2,4);
ft_write_mri(sprintf('%sFig3/Middle_D2_Tstat.nii',results_dir),image,...
    'dataformat','nifti');
image.anatomy = mean(middle_Tstat_D5,4);
ft_write_mri(sprintf('%sFig3/Middle_D5_Tstat.nii',results_dir),image,...
    'dataformat','nifti');

image.anatomy = mean(oldest_Tstat_D2,4) ;
ft_write_mri(sprintf('%sFig3/Oldest_D2_Tstat.nii',results_dir),image,...
    'dataformat','nifti');
image.anatomy = mean(oldest_Tstat_D5,4);
ft_write_mri(sprintf('%sFig3/Oldest_D5_Tstat.nii',results_dir),image,...
    'dataformat','nifti');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Burst Spectra
youngest_hmm_spec_D2 = hmm_results_kids.burst_spec_D1(youngest,:);
youngest_hmm_spec_D5 = hmm_results_kids.burst_spec_D4(youngest,:);
middle_hmm_spec_D2 = hmm_results_kids.burst_spec_D1(middle,:);
middle_hmm_spec_D5 = hmm_results_kids.burst_spec_D4(middle,:);
oldest_hmm_spec_D2 = hmm_results_kids.burst_spec_D1(oldest,:);
oldest_hmm_spec_D5 = hmm_results_kids.burst_spec_D4(oldest,:);
    
youngest_hmm_spec_D2_ad = hmm_results_adults.burst_spec_D1(youngest_ad,:);
youngest_hmm_spec_D5_ad = hmm_results_adults.burst_spec_D4(youngest_ad,:);
middle_hmm_spec_D2_ad = hmm_results_adults.burst_spec_D1(middle_ad,:);
middle_hmm_spec_D5_ad = hmm_results_adults.burst_spec_D4(middle_ad,:);
oldest_hmm_spec_D2_ad = hmm_results_adults.burst_spec_D1(oldest_ad,:);
oldest_hmm_spec_D5_ad = hmm_results_adults.burst_spec_D4(oldest_ad,:);
all_spec_D2 = [hmm_results_kids.burst_spec_D1;hmm_results_adults.burst_spec_D1];
all_spec_D5 = [hmm_results_kids.burst_spec_D4;hmm_results_adults.burst_spec_D4];
load hmm_fs.mat hmm_fs
all_ages = [kids_ages;adult_ages];

for fi = 1:size(all_spec_D2,2)
[corr_spec_D2(fi),p_spec_D2(fi)] = corr(all_ages,all_spec_D2(:,fi));
[corr_spec_D5(fi),p_spec_D5(fi)] = corr(all_ages,all_spec_D5(:,fi));
end

xlineinds = [26,87,210,374];
xlines = [hmm_fs(xlineinds(1)),hmm_fs(xlineinds(2)),...
    hmm_fs(xlineinds(3)),hmm_fs(xlineinds(4))]; % ~ 3,9,21,27 Hz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% D2
f_spec_D2 = figure('Color','w', ...
    'Name','HMM SPectra D2', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
% cols = [161,218,180;
% 65,182,196;
% 44,127,184;
% 37,52,148]./255;
cols = lines(6);
lw = 1;

ax_D2 = subplot(1,1,1);
hold on

m = mean(hmm_results_adults.burst_spec_D1,1);
err = std(hmm_results_adults.burst_spec_D1,[],1)./sqrt(size(hmm_results_adults.burst_spec_D1,1));
c(4) = ciplot(m-err,m+err,hmm_fs,cols(4,:)); c(4).DisplayName = 'S.E. Adults';

m = mean(youngest_hmm_spec_D2,1);
err = std(youngest_hmm_spec_D2,[],1)./sqrt(size(youngest_hmm_spec_D2,1));
c(1) = ciplot(m-err,m+err,hmm_fs,cols(1,:)); c(1).DisplayName = 'S.E. Youngest';

m = mean(middle_hmm_spec_D2,1);
err = std(middle_hmm_spec_D2,[],1)./sqrt(size(middle_hmm_spec_D2,1));
c(2) = ciplot(m-err,m+err,hmm_fs,cols(2,:)); c(2).DisplayName = 'S.E. Middle';

m = mean(oldest_hmm_spec_D2,1);
err = std(oldest_hmm_spec_D2,[],1)./sqrt(size(oldest_hmm_spec_D2,1));
c(3) = ciplot(m-err,m+err,hmm_fs,cols(3,:)); c(3).DisplayName = 'S.E. Oldest';


p_adults = plot(hmm_fs,mean(hmm_results_adults.burst_spec_D1,1), ...
    'Color',cols(4,:),'Linewidth',lw,'DisplayName','Adults');

p_y = plot(hmm_fs,mean(youngest_hmm_spec_D2,1), ...
    'Color',cols(1,:),'Linewidth',lw,'DisplayName','Youngest');
p_m = plot(hmm_fs,mean(middle_hmm_spec_D2,1), ...
    'Color', cols(2,:),'Linewidth',lw,'DisplayName','Middle');
p_o = plot(hmm_fs,mean(oldest_hmm_spec_D2,1), ...
    'Color', cols(3,:),'Linewidth',lw,'DisplayName','Oldest');
xlim([0,50])
ylim([0,0.1])
xlabel('Frequency (Hz)')
ylabel('State PSD')

set(c,'Handlevisibility','off')
set(ax_D2,'XScale','linear','YScale','linear')

% for li=1:length(xlines)
%     xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
% end

set(f_spec_D2.Children,'FontSize',9)
legend([p_y,p_m,p_o,p_adults],'Location','best')
axis fill
print(f_spec_D2,sprintf('%sFig6/Spectra_D2.png',results_dir),'-dpng','-r900');


%%% age corr spectra

f_spec_D2 = figure('Color','w', ...
    'Name','HMM SPectra corr D2', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
ax_D2 = subplot(1,1,1);

alpha_ = 0.01;

plot(hmm_fs,corr_spec_D2,'HandleVisibility','off','LineWidth',2)
xlabel('Frequency (Hz)');
ylabel('R (Power vs. Age)');hold on

ylims = ylim;
y_ = p_spec_D2<alpha_;y=ones(size(y_));y(y_== 0) = ylims(1); y(y_== 1) = ylims(2);
shading = area(hmm_fs,y,ylims(1),'FaceColor','r','EdgeColor','none','FaceAlpha',0.2,...
    'DisplayName',sprintf('p < %1.2f',alpha_));

yline(0,'HandleVisibility','off')

xlim([0,50])

for li=1:length(xlines)
    xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
end
axis fill
set(f_spec_D2.Children,'FontSize',9)
legend('Location','best')

print(f_spec_D2,sprintf('%sFig6/Spectra_v_age_D2.png',results_dir),'-dpng','-r900');

txt_locs = [1,0.01;...
            15,0.1;...
            1,0.05;...
            1,2.5E-3 ];
%%%%%%%%% scatters
for li=1:length(xlines)
    fi =  xlineinds(li);

    f_spec_corr = figure('Name','Spec v age single f', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
    fwidth = 5;
    fheight = 4.8;
    f_spec_corr.Position([3,4]) = [fwidth,fheight];

    scatter(kids_ages,hmm_results_kids.burst_spec_D1(:,fi),'k^');hold on
    scatter(adult_ages,hmm_results_adults.burst_spec_D1(:,fi),'ko')

    ylabel('PSD (A.U.)')
    xlabel('Age (years)')
    xlim([0,1+max(adult_ages)])
%     ylim([-0.2,0.9])

    x = all_ages;
    X = [ones(length(x),1) x];
    y = all_spec_D2(:,fi);
    b = X\y;
    [R_spec_D2,p_spec_D2] = corr(x,y);
    refl = refline(b(2),b(1));

    text(txt_locs(li,1),txt_locs(li,2),sprintf('R^2 = %1.1f\np = %1.2e',R_spec_D2^2,p_spec_D2))
    set(f_spec_corr.Children,'FontName','Arial','FontSize',9)
    print(f_spec_corr,sprintf('%sFig6/spec_v_age_D2_%1.0fHz.png',results_dir, xlines(li)),'-dpng','-r900');
end

%%% D5
f_spec_D5 = figure('Color','w', ...
    'Name','HMM SPectra D5', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
% cols = [161,218,180;
% 65,182,196;
% 44,127,184;
% 37,52,148]./255;
cols = lines(6);
lw = 1;

ax_D5 = subplot(1,1,1);
hold on

m = mean(hmm_results_adults.burst_spec_D4,1);
err = std(hmm_results_adults.burst_spec_D4,[],1)./sqrt(size(hmm_results_adults.burst_spec_D4,1));
c(4) = ciplot(m-err,m+err,hmm_fs,cols(4,:)); c(4).DisplayName = 'S.E. Adults';

m = mean(youngest_hmm_spec_D5,1);
err = std(youngest_hmm_spec_D5,[],1)./sqrt(size(youngest_hmm_spec_D5,1));
c(1) = ciplot(m-err,m+err,hmm_fs,cols(1,:)); c(1).DisplayName = 'S.E. Youngest';

m = mean(middle_hmm_spec_D5,1);
err = std(middle_hmm_spec_D5,[],1)./sqrt(size(middle_hmm_spec_D5,1));
c(2) = ciplot(m-err,m+err,hmm_fs,cols(2,:)); c(2).DisplayName = 'S.E. Middle';

m = mean(oldest_hmm_spec_D5,1);
err = std(oldest_hmm_spec_D5,[],1)./sqrt(size(oldest_hmm_spec_D5,1));
c(3) = ciplot(m-err,m+err,hmm_fs,cols(3,:)); c(3).DisplayName = 'S.E. Oldest';


p_adults = plot(hmm_fs,mean(hmm_results_adults.burst_spec_D4,1), ...
    'Color',cols(4,:),'Linewidth',lw,'DisplayName','Adults');

p_y = plot(hmm_fs,mean(youngest_hmm_spec_D5,1), ...
    'Color',cols(1,:),'Linewidth',lw,'DisplayName','Youngest');
p_m = plot(hmm_fs,mean(middle_hmm_spec_D5,1), ...
    'Color', cols(2,:),'Linewidth',lw,'DisplayName','Middle');
p_o = plot(hmm_fs,mean(oldest_hmm_spec_D5,1), ...
    'Color', cols(3,:),'Linewidth',lw,'DisplayName','Oldest');
xlim([0,50])
ylim([0,0.1])
xlabel('Frequency (Hz)')
ylabel('State PSD')

set(c,'Handlevisibility','off')
set(ax_D5,'XScale','linear','YScale','linear')

% for li=1:length(xlines)
%     xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
% end

set(f_spec_D5.Children,'FontSize',9)
legend([p_y,p_m,p_o,p_adults],'Location','best')
axis fill
print(f_spec_D5,sprintf('%sFig6/Spectra_D5.png',results_dir),'-dpng','-r900');
%%% age corr spectra

f_spec_D5 = figure('Color','w', ...
    'Name','HMM SPectra corr D2', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
ax_D5 = subplot(1,1,1);

plot(hmm_fs,corr_spec_D5,'HandleVisibility','off','LineWidth',2)
xlabel('Frequency (Hz)');
ylabel('R (Power vs. Age)');hold on

ylims = ylim;
y_ = p_spec_D5<alpha_;y=ones(size(y_));y(y_== 0) = ylims(1); y(y_== 1) = ylims(2);
shading = area(hmm_fs,y,ylims(1),'FaceColor','r','EdgeColor','none','FaceAlpha',0.2,...
    'DisplayName',sprintf('p < %1.2f',alpha_));

yline(0,'HandleVisibility','off')

xlim([0,50])

for li=1:length(xlines)
    xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
end
axis fill
set(f_spec_D5.Children,'FontSize',9)
legend('Location','best')

print(f_spec_D5,sprintf('%sFig6/Spectra_v_ageD5.png',results_dir),'-dpng','-r900');

txt_locs = [1,0.015;...
            15,0.15;...
            1,0.05;...
            1,2.5E-3 ];

%%%%% scatters D5
for li=1:length(xlines)
    fi =  xlineinds(li);

    f_spec_corr = figure('Name','Spec v age single f', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
    fwidth = 5;
    fheight = 4.8;
    f_spec_corr.Position([3,4]) = [fwidth,fheight];

    scatter(kids_ages,hmm_results_kids.burst_spec_D4(:,fi),'k^');hold on
    scatter(adult_ages,hmm_results_adults.burst_spec_D4(:,fi),'ko')

    ylabel('PSD (A.U.)')
    xlabel('Age (years)')
    xlim([0,1+max(adult_ages)])
%     ylim([-0.2,0.9])

    x = all_ages;
    X = [ones(length(x),1) x];
    y = all_spec_D5(:,fi);
    b = X\y;
    [R_spec_D5,p_spec_D5] = corr(x,y);
    refl = refline(b(2),b(1));

    text(txt_locs(li,1),txt_locs(li,2),sprintf('R^2 = %1.1f\np = %1.2e',R_spec_D5^2,p_spec_D5))
    set(f_spec_corr.Children,'FontName','Arial','FontSize',9)
    print(f_spec_corr,sprintf('%sFig6/spec_v_age_D5_%1.0fHz.png',results_dir, xlines(li)),'-dpng','-r900');
end


%% NON Burst Spectra
youngest_non_burst_hmm_spec_D2 = hmm_results_kids.non_burst_spec_D1(youngest,:);
youngest_non_burst_hmm_spec_D5 = hmm_results_kids.non_burst_spec_D4(youngest,:);
middle_non_burst_hmm_spec_D2 = hmm_results_kids.non_burst_spec_D1(middle,:);
middle_non_burst_hmm_spec_D5 = hmm_results_kids.non_burst_spec_D4(middle,:);
oldest_non_burst_hmm_spec_D2 = hmm_results_kids.non_burst_spec_D1(oldest,:);
oldest_non_burst_hmm_spec_D5 = hmm_results_kids.non_burst_spec_D4(oldest,:);
    
youngest_non_burst_hmm_spec_D2_ad = hmm_results_adults.non_burst_spec_D1(youngest_ad,:);
youngest_non_burst_hmm_spec_D5_ad = hmm_results_adults.non_burst_spec_D4(youngest_ad,:);
middle_non_burst_hmm_spec_D2_ad = hmm_results_adults.non_burst_spec_D1(middle_ad,:);
middle_non_burst_hmm_spec_D5_ad = hmm_results_adults.non_burst_spec_D4(middle_ad,:);
oldest_non_burst_hmm_spec_D2_ad = hmm_results_adults.non_burst_spec_D1(oldest_ad,:);
oldest_non_burst_hmm_spec_D5_ad = hmm_results_adults.non_burst_spec_D4(oldest_ad,:);
all_non_burst_spec_D2 = [hmm_results_kids.non_burst_spec_D1;hmm_results_adults.non_burst_spec_D1];
all_non_burst_spec_D5 = [hmm_results_kids.non_burst_spec_D4;hmm_results_adults.non_burst_spec_D4];
load hmm_fs.mat hmm_fs
all_ages = [kids_ages;adult_ages];

for fi = 1:size(all_non_burst_spec_D2,2)
[corr_non_b_spec_D2(fi),p_non_b_spec_D2(fi)] = corr(all_ages,all_non_burst_spec_D2(:,fi));
[corr_non_b_spec_D5(fi),p_non_b_spec_D5(fi)] = corr(all_ages,all_non_burst_spec_D5(:,fi));
end

xlineinds = [26,87,210,374];
xlines = [hmm_fs(xlineinds(1)),hmm_fs(xlineinds(2)),...
    hmm_fs(xlineinds(3)),hmm_fs(xlineinds(4))]; % ~ 3,9,21,27 Hz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% D2
f_non_b_spec_D2 = figure('Color','w', ...
    'Name','HMM Non Burst SPectra D2', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
% cols = [161,218,180;
% 65,182,196;
% 44,127,184;
% 37,52,148]./255;
cols = lines(6);
lw = 1;

ax_D2_non_b = subplot(1,1,1);
hold on

m = mean(hmm_results_adults.non_burst_spec_D1,1);
err = std(hmm_results_adults.non_burst_spec_D1,[],1)./sqrt(size(hmm_results_adults.non_burst_spec_D1,1));
c(4) = ciplot(m-err,m+err,hmm_fs,cols(4,:)); c(4).DisplayName = 'S.E. Adults';

m = mean(youngest_non_burst_hmm_spec_D2,1);
err = std(youngest_non_burst_hmm_spec_D2,[],1)./sqrt(size(youngest_non_burst_hmm_spec_D2,1));
c(1) = ciplot(m-err,m+err,hmm_fs,cols(1,:)); c(1).DisplayName = 'S.E. Youngest';

m = mean(middle_non_burst_hmm_spec_D2,1);
err = std(middle_non_burst_hmm_spec_D2,[],1)./sqrt(size(middle_non_burst_hmm_spec_D2,1));
c(2) = ciplot(m-err,m+err,hmm_fs,cols(2,:)); c(2).DisplayName = 'S.E. Middle';

m = mean(oldest_non_burst_hmm_spec_D2,1);
err = std(oldest_non_burst_hmm_spec_D2,[],1)./sqrt(size(oldest_non_burst_hmm_spec_D2,1));
c(3) = ciplot(m-err,m+err,hmm_fs,cols(3,:)); c(3).DisplayName = 'S.E. Oldest';


p_adults = plot(hmm_fs,mean(hmm_results_adults.non_burst_spec_D1,1), ...
    'Color',cols(4,:),'Linewidth',lw,'DisplayName','Adults');

p_y = plot(hmm_fs,mean(youngest_non_burst_hmm_spec_D2,1), ...
    'Color',cols(1,:),'Linewidth',lw,'DisplayName','Youngest');
p_m = plot(hmm_fs,mean(middle_non_burst_hmm_spec_D2,1), ...
    'Color', cols(2,:),'Linewidth',lw,'DisplayName','Middle');
p_o = plot(hmm_fs,mean(oldest_non_burst_hmm_spec_D2,1), ...
    'Color', cols(3,:),'Linewidth',lw,'DisplayName','Oldest');
xlim([0,50])
ylim([0,0.1])
xlabel('Frequency (Hz)')
ylabel('State PSD')

set(c,'Handlevisibility','off')
set(ax_D2_non_b,'XScale','linear','YScale','linear')

% for li=1:length(xlines)
%     xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
% end

set(f_non_b_spec_D2.Children,'FontSize',9)
legend([p_y,p_m,p_o,p_adults],'Location','best')
axis fill

print(f_non_b_spec_D2,sprintf('%sFig6/Non_b_Spectra_D2.png',results_dir),'-dpng','-r900');

%%% age corr spectra
f_non_b_spec_D2 = figure('Color','w', ...
    'Name','HMM SPectra corr D2', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
ax_D2 = subplot(1,1,1);

alpha_ = 0.01;

plot(hmm_fs,corr_non_b_spec_D2,'HandleVisibility','off','LineWidth',2)
xlabel('Frequency (Hz)');
ylabel('R (Power vs. Age)');hold on

ylims = ylim;
y_ = p_non_b_spec_D2<alpha_;y=ones(size(y_));y(y_== 0) = ylims(1); y(y_== 1) = ylims(2);
shading = area(hmm_fs,y,ylims(1),'FaceColor','r','EdgeColor','none','FaceAlpha',0.2,...
    'DisplayName',sprintf('p < %1.2f',alpha_));

yline(0,'HandleVisibility','off')

xlim([0,50])

for li=1:length(xlines)
    xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
end
axis fill
set(f_non_b_spec_D2.Children,'FontSize',9)
legend('Location','best')

print(f_non_b_spec_D2,sprintf('%sFig6/non_b_Spectra_v_Age_D2.png',results_dir),'-dpng','-r900');

txt_locs = [1,0.012;...
            1,0.05;...
            1,0.025;...
            1,2.5E-3];
%%%%%%%%% scatters
for li=1:length(xlines)
    fi =  xlineinds(li);

    f_non_b_spec_corr = figure('Name','Non b. Spec v age single f', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
    fwidth = 5;
    fheight = 4.8;
    f_non_b_spec_corr.Position([3,4]) = [fwidth,fheight];

 
    scatter(kids_ages,hmm_results_kids.non_burst_spec_D1(:,fi),'k^');hold on
    scatter(adult_ages,hmm_results_adults.non_burst_spec_D1(:,fi),'ko')

    ylabel('PSD (A.U.)')
    xlabel('Age (years)')
    xlim([0,1+max(adult_ages)])
%     ylim([-0.2,0.9])

    x = all_ages;
    X = [ones(length(x),1) x];
    y = all_non_burst_spec_D2(:,fi);
    b = X\y;
    [R_non_b_spec_D2,p_non_b_spec_D2] = corr(x,y);
    refl = refline(b(2),b(1));

    text(txt_locs(li,1),txt_locs(li,2),sprintf('R^2 = %1.1f\np = %1.2e',R_non_b_spec_D2^2,p_non_b_spec_D2))
    set(f_non_b_spec_corr.Children,'FontName','Arial','FontSize',9)
    print(f_non_b_spec_corr,sprintf('%sFig6/non_b_spec_v_age_D2_%1.0fHz.png',results_dir, xlines(li)),'-dpng','-r900');
end

%%% D5
f_non_burst_spec_D5 = figure('Color','w', ...
    'Name','Non b. HMM SPectra D5', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
% cols = [161,218,180;
% 65,182,196;
% 44,127,184;
% 37,52,148]./255;
cols = lines(6);
lw = 1;

ax_D5_non_b = subplot(1,1,1);
hold on

m = mean(hmm_results_adults.non_burst_spec_D4,1);
err = std(hmm_results_adults.non_burst_spec_D4,[],1)./sqrt(size(hmm_results_adults.non_burst_spec_D4,1));
c(4) = ciplot(m-err,m+err,hmm_fs,cols(4,:)); c(4).DisplayName = 'S.E. Adults';

m = mean(youngest_non_burst_hmm_spec_D5,1);
err = std(youngest_non_burst_hmm_spec_D5,[],1)./sqrt(size(youngest_non_burst_hmm_spec_D5,1));
c(1) = ciplot(m-err,m+err,hmm_fs,cols(1,:)); c(1).DisplayName = 'S.E. Youngest';

m = mean(middle_non_burst_hmm_spec_D5,1);
err = std(middle_non_burst_hmm_spec_D5,[],1)./sqrt(size(middle_non_burst_hmm_spec_D5,1));
c(2) = ciplot(m-err,m+err,hmm_fs,cols(2,:)); c(2).DisplayName = 'S.E. Middle';

m = mean(oldest_non_burst_hmm_spec_D5,1);
err = std(oldest_non_burst_hmm_spec_D5,[],1)./sqrt(size(oldest_non_burst_hmm_spec_D5,1));
c(3) = ciplot(m-err,m+err,hmm_fs,cols(3,:)); c(3).DisplayName = 'S.E. Oldest';


p_adults = plot(hmm_fs,mean(hmm_results_adults.non_burst_spec_D4,1), ...
    'Color',cols(4,:),'Linewidth',lw,'DisplayName','Adults');

p_y = plot(hmm_fs,mean(youngest_non_burst_hmm_spec_D5,1), ...
    'Color',cols(1,:),'Linewidth',lw,'DisplayName','Youngest');
p_m = plot(hmm_fs,mean(middle_non_burst_hmm_spec_D5,1), ...
    'Color', cols(2,:),'Linewidth',lw,'DisplayName','Middle');
p_o = plot(hmm_fs,mean(oldest_non_burst_hmm_spec_D5,1), ...
    'Color', cols(3,:),'Linewidth',lw,'DisplayName','Oldest');
xlim([0,50])
ylim([0,0.1])
xlabel('Frequency (Hz)')
ylabel('State PSD')

set(c,'Handlevisibility','off')
set(ax_D5_non_b,'XScale','linear','YScale','linear')

% for li=1:length(xlines)
%     xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
% end

set(f_non_burst_spec_D5.Children,'FontSize',9)
legend([p_y,p_m,p_o,p_adults],'Location','best')
axis fill

print(f_non_burst_spec_D5,sprintf('%sFig6/Non_b_Spectra_D5.png',results_dir),'-dpng','-r900');

%%% age corr spectra

f_non_b_spec_D5 = figure('Color','w', ...
    'Name','HMM SPectra corr D2', ...
    'Units','centimeters', ...
    'Position',[4,4,19.8,5]);
ax_D5 = subplot(1,1,1);

plot(hmm_fs,corr_non_b_spec_D5,'HandleVisibility','off','LineWidth',2)
xlabel('Frequency (Hz)');
ylabel('R (Power vs. Age)');hold on

ylims = ylim;
y_ = p_non_b_spec_D5<alpha_;y=ones(size(y_));y(y_== 0) = ylims(1); y(y_== 1) = ylims(2);
shading = area(hmm_fs,y,ylims(1),'FaceColor','r','EdgeColor','none','FaceAlpha',0.2,...
    'DisplayName',sprintf('p < %1.2f',alpha_));

yline(0,'HandleVisibility','off')

xlim([0,50])

for li=1:length(xlines)
    xline(xlines(li),'k','LineWidth',2,'HandleVisibility','off')
end
axis fill
set(f_non_burst_spec_D5.Children,'FontSize',9)
legend('Location','best')

print(f_non_b_spec_D5,sprintf('%sFig6/non_b_Spectra_v_age_D5.png',results_dir),'-dpng','-r900');

txt_locs = [1,0.015;...
            1,0.06;...
            1,0.025;...
            1,2.5E-3];
%scatters
for li=1:length(xlines)
    fi =  xlineinds(li);

    f_non_b_spec_corr = figure('Name','Non b. Spec v age single f', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
    fwidth = 5;
    fheight = 4.8;
    f_non_b_spec_corr.Position([3,4]) = [fwidth,fheight];

    scatter(kids_ages,hmm_results_kids.non_burst_spec_D4(:,fi),'k^');hold on
    scatter(adult_ages,hmm_results_adults.non_burst_spec_D4(:,fi),'ko')

    ylabel('PSD (A.U.)')
    xlabel('Age (years)')
    xlim([0,1+max(adult_ages)])
%     ylim([-0.2,0.9])

    x = all_ages;
    X = [ones(length(x),1) x];
    y = all_non_burst_spec_D5(:,fi);
    b = X\y;
    [R_non_b_spec_D5,p_non_b_spec_D5] = corr(x,y);
    refl = refline(b(2),b(1));

    text(txt_locs(li,1),txt_locs(li,2),sprintf('R^2 = %1.1f\np = %1.2e',R_non_b_spec_D5^2,p_non_b_spec_D5))
    set(f_non_b_spec_corr.Children,'FontName','Arial','FontSize',9)
    print(f_non_b_spec_corr,sprintf('%sFig6/non_b_spec_v_age_D5_%1.0fHz.png',results_dir, xlines(li)),'-dpng','-r900');
end
%% raster
youngest_hmm_raster_D2 = hmm_results_kids.burst_raster_D1(youngest,:);
youngest_hmm_raster_D5 = hmm_results_kids.burst_raster_D4(youngest,:);
middle_hmm_raster_D2 = hmm_results_kids.burst_raster_D1(middle,:);
middle_hmm_raster_D5 = hmm_results_kids.burst_raster_D4(middle,:);
oldest_hmm_raster_D2 = hmm_results_kids.burst_raster_D1(oldest,:);
oldest_hmm_raster_D5 = hmm_results_kids.burst_raster_D4(oldest,:);
    
youngest_hmm_raster_D2_ad = hmm_results_adults.burst_raster_D1(youngest_ad,:);
youngest_hmm_raster_D5_ad = hmm_results_adults.burst_raster_D4(youngest_ad,:);
middle_hmm_raster_D2_ad = hmm_results_adults.burst_raster_D1(middle_ad,:);
middle_hmm_raster_D5_ad = hmm_results_adults.burst_raster_D4(middle_ad,:);
oldest_hmm_raster_D2_ad = hmm_results_adults.burst_raster_D1(oldest_ad,:);
oldest_hmm_raster_D5_ad = hmm_results_adults.burst_raster_D4(oldest_ad,:);


f = figure('Color','w','Units','centimeters','Name','D2 raster');
% f.Position([3,4]) = [40,10];
subplot(3,2,1)
plot_raster(youngest_hmm_raster_D2,[])
title('Youngest')

subplot(3,2,3)
plot_raster(middle_hmm_raster_D2,[])
title('Middle')

subplot(3,2,5)
plot_raster(oldest_hmm_raster_D2,[])
title('Oldest')


subplot(3,2,2)
plot_raster(youngest_hmm_raster_D2_ad,[])
title('Adults 1')
subplot(3,2,4)
plot_raster(middle_hmm_raster_D2_ad,[])
title('Adults 2')
subplot(3,2,6)
plot_raster(oldest_hmm_raster_D2_ad,[])
title('Adults 3')

sgtitle('D2')
set(gcf,'Position',[32.8348 4.0217 14.8167 21.5900])

%%%%
f = figure('Color','w','Units','centimeters','Name','D5 raster');
% f.Position([3,4]) = [40,10];
cbar_lim = 0.3;
subplot(3,2,1)
plot_raster(youngest_hmm_raster_D5,[])
title('Youngest')

subplot(3,2,3)
plot_raster(middle_hmm_raster_D5,[])
title('Middle')

subplot(3,2,5)
plot_raster(oldest_hmm_raster_D5,[])
title('Oldest')


subplot(3,2,2)
plot_raster(youngest_hmm_raster_D5_ad,[])
title('Adults 1')
subplot(3,2,4)
plot_raster(middle_hmm_raster_D5_ad,[])
title('Adults 2')
subplot(3,2,6)
plot_raster(oldest_hmm_raster_D5_ad,[])
title('Adults 3')
sgtitle('D5')
set(gcf,'Position',[17.9652 4.1275 14.8167 21.7223])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% PAPER FIG 5
%% combined raster plot
hmm_results_kids_srt = sortrows(hmm_results_kids,"age","ascend");
hmm_results_adults_srt = sortrows(hmm_results_adults,"age","ascend");

grpsz = [length(youngest),length(middle),length(oldest),length(adult_ages)];

fr1 = figure('Name','Combined raster D2','Units','centimeters','Color','w');
plot_raster([hmm_results_kids_srt.burst_raster_D1;hmm_results_adults_srt.burst_raster_D1], ...
grpsz)
fwidth = 7.8;
fheight = 20.5;
fr1.Position([3,4]) = [fwidth,fheight];
axis fill
fr1.Children(1).Children(1).DisplayName = 'Group boundary';
fr1.Children(1).Children(2).DisplayName = 'Subject boundary';
set(fr1.Children(1).Children(3:end),'HandleVisibility','off');
legend('Location','Northoutside')
set(fr1.Children,'FontSize',9)
fr1.Children(2).TickDir = 'out';
print(fr1,sprintf('%sFig5/Raster_All_D2.png',results_dir),'-dpng','-r900');


fr2 = figure('Name','Combined raster D5','Units','centimeters','Color','w');
plot_raster([hmm_results_kids_srt.burst_raster_D4;hmm_results_adults_srt.burst_raster_D4], ...
    grpsz)
fr2.Position([3,4]) = [fwidth,fheight];
axis fill
fr2.Children(1).Children(1).DisplayName = 'Group boundary';
fr2.Children(1).Children(2).DisplayName = 'Subject boundary';
set(fr2.Children(1).Children(3:end),'HandleVisibility','off');
legend('Location','Northoutside')
set(fr2.Children,'FontSize',9)
fr2.Children(2).TickDir = 'out';
print(fr2,sprintf('%sFig5/Raster_All_D5.png',results_dir),'-dpng','-r900');

%%
addpath('C:\Users\ppynr2\OneDrive - The University of Nottingham\Documents\GitHub\Braille_kids\Violinplot-Matlab-master')
addpath('.\Violinplot-Matlab-master')
cmean = @(mat) mean(mat,1);

youngest_hmm_burst_prob_D2 = cell2mat(cellfun(cmean,youngest_hmm_raster_D2,'UniformOutput',false));
youngest_hmm_burst_prob_D5 = cell2mat(cellfun(cmean,youngest_hmm_raster_D5,'UniformOutput',false));
middle_hmm_burst_prob_D2 = cell2mat(cellfun(cmean,middle_hmm_raster_D2,'UniformOutput',false));
middle_hmm_burst_prob_D5 = cell2mat(cellfun(cmean,middle_hmm_raster_D5,'UniformOutput',false));
oldest_hmm_burst_prob_D2 = cell2mat(cellfun(cmean,oldest_hmm_raster_D2,'UniformOutput',false));
oldest_hmm_burst_prob_D5 = cell2mat(cellfun(cmean,oldest_hmm_raster_D5 ,'UniformOutput',false));
    
youngest_hmm_burst_prob_D2_ad = cell2mat(cellfun(cmean,youngest_hmm_raster_D2_ad,'UniformOutput',false));
youngest_hmm_burst_prob_D5_ad = cell2mat(cellfun(cmean,youngest_hmm_raster_D5_ad,'UniformOutput',false));
middle_hmm_burst_prob_D2_ad = cell2mat(cellfun(cmean,middle_hmm_raster_D2_ad,'UniformOutput',false));
middle_hmm_burst_prob_D5_ad = cell2mat(cellfun(cmean,middle_hmm_raster_D5_ad,'UniformOutput',false));
oldest_hmm_burst_prob_D2_ad = cell2mat(cellfun(cmean,oldest_hmm_raster_D2_ad,'UniformOutput',false));
oldest_hmm_burst_prob_D5_ad = cell2mat(cellfun(cmean,oldest_hmm_raster_D5_ad,'UniformOutput',false));

fprob = figure('Name','Prob time courses grp D2','Color','w','Units','centimeters');
fheight = 6.5;
fwidth = 10.2;
fprob.Position([3,4]) = [fwidth,fheight];
trl_t = linspace(0,3.5,350);
hold on
ph(1) = plot(trl_t,mean(youngest_hmm_burst_prob_D2),'DisplayName','Children_Y');
ph(2) = plot(trl_t,mean(middle_hmm_burst_prob_D2),'DisplayName','Children_M');
ph(3) = plot(trl_t,mean(oldest_hmm_burst_prob_D2),'DisplayName','Children_O');
ph(4) = plot(trl_t,mean([youngest_hmm_burst_prob_D2_ad; ...
    middle_hmm_burst_prob_D2_ad; ...
    oldest_hmm_burst_prob_D2_ad]),'DisplayName','Adults');
m = mean(youngest_hmm_burst_prob_D2);
err = std(youngest_hmm_burst_prob_D2,[],1)./sqrt(size(youngest_hmm_burst_prob_D2,1));
ch(1) = ciplot(m-err,m+err,trl_t,ph(1).Color); ch(1).HandleVisibility = 'off';

m = mean(middle_hmm_burst_prob_D2);
err = std(middle_hmm_burst_prob_D2,[],1)./sqrt(size(middle_hmm_burst_prob_D2,1));
ch(2) = ciplot(m-err,m+err,trl_t,ph(2).Color); ch(2).HandleVisibility = 'off';

m = mean(oldest_hmm_burst_prob_D2);
err = std(oldest_hmm_burst_prob_D2,[],1)./sqrt(size(oldest_hmm_burst_prob_D2,1));
ch(3) = ciplot(m-err,m+err,trl_t,ph(3).Color); ch(3).HandleVisibility = 'off';

m = mean([youngest_hmm_burst_prob_D2_ad; ...
    middle_hmm_burst_prob_D2_ad; ...
    oldest_hmm_burst_prob_D2_ad]);
err = std([youngest_hmm_burst_prob_D2_ad; ...
    middle_hmm_burst_prob_D2_ad; ...
    oldest_hmm_burst_prob_D2_ad],[],1)./sqrt(size([youngest_hmm_burst_prob_D2_ad; ...
    middle_hmm_burst_prob_D2_ad; ...
    oldest_hmm_burst_prob_D2_ad],1));
ch(4) = ciplot(m-err,m+err,trl_t,ph(4).Color); ch(4).HandleVisibility = 'off';
ylabel('Burst probability (A.U.)')
xlabel('time (s)')
axis fill
legend('Location','southeast','NumColumns',2)
set(ph, 'Linewidth', 1.5)
set(fprob.Children, 'FontSize', 9)
fprob.Children(2).TickDir = 'out';
xlim([0,3.5])
ylim([0,0.5])
axx=gca;axx.Position(2)=0.17;
print(fprob,sprintf('%sFig5/prob_tc_D2.png',results_dir),'-dpng','-r900');

fprob = figure('Name','Prob time courses grp D5','Color','w','Units','centimeters');
fheight = 6.5;
fwidth = 10.2;
fprob.Position([3,4]) = [fwidth,fheight];
trl_t = linspace(0,3.5,350);
hold on
ph(1) = plot(trl_t,mean(youngest_hmm_burst_prob_D5),'DisplayName','Children_Y');
ph(2) = plot(trl_t,mean(middle_hmm_burst_prob_D5),'DisplayName','Children_M');
ph(3) = plot(trl_t,mean(oldest_hmm_burst_prob_D5),'DisplayName','Children_O');
ph(4) = plot(trl_t,mean([youngest_hmm_burst_prob_D5_ad; ...
    middle_hmm_burst_prob_D5_ad; ...
    oldest_hmm_burst_prob_D5_ad]),'DisplayName','Adults');
m = mean(youngest_hmm_burst_prob_D5);
err = std(youngest_hmm_burst_prob_D5,[],1)./sqrt(size(youngest_hmm_burst_prob_D5,1));
ch(1) = ciplot(m-err,m+err,trl_t,ph(1).Color); ch(1).HandleVisibility = 'off';

m = mean(middle_hmm_burst_prob_D5);
err = std(middle_hmm_burst_prob_D5,[],1)./sqrt(size(middle_hmm_burst_prob_D5,1));
ch(2) = ciplot(m-err,m+err,trl_t,ph(2).Color); ch(2).HandleVisibility = 'off';

m = mean(oldest_hmm_burst_prob_D5);
err = std(oldest_hmm_burst_prob_D5,[],1)./sqrt(size(oldest_hmm_burst_prob_D5,1));
ch(3) = ciplot(m-err,m+err,trl_t,ph(3).Color); ch(3).HandleVisibility = 'off';

m = mean([youngest_hmm_burst_prob_D5_ad; ...
    middle_hmm_burst_prob_D5_ad; ...
    oldest_hmm_burst_prob_D5_ad]);
err = std([youngest_hmm_burst_prob_D5_ad; ...
    middle_hmm_burst_prob_D5_ad; ...
    oldest_hmm_burst_prob_D5_ad],[],1)./sqrt(size([youngest_hmm_burst_prob_D5_ad; ...
    middle_hmm_burst_prob_D5_ad; ...
    oldest_hmm_burst_prob_D5_ad],1));
ch(4) = ciplot(m-err,m+err,trl_t,ph(4).Color); ch(4).HandleVisibility = 'off';
ylabel('Burst probability (A.U.)')
xlabel('time (s)')
axis fill
legend('Location','southeast','NumColumns',2)
set(ph, 'Linewidth', 1.5)
set(fprob.Children, 'FontSize', 9)
fprob.Children(2).TickDir = 'out';
xlim([0,3.5])
ylim([0,0.5])
axx=gca;axx.Position(2)=0.17;
print(fprob,sprintf('%sFig5/prob_tc_D5.png',results_dir),'-dpng','-r900');



prob_D2 = [cell2mat(cellfun(cmean,hmm_results_kids.burst_raster_D1,'UniformOutput',false));...
    cell2mat(cellfun(cmean,hmm_results_adults.burst_raster_D1,'UniformOutput',false))];
prob_D5 = [cell2mat(cellfun(cmean,hmm_results_kids.burst_raster_D4,'UniformOutput',false));...
    cell2mat(cellfun(cmean,hmm_results_adults.burst_raster_D4,'UniformOutput',false))];

for ti = 1:size(prob_D2,2)
[corrc_D2(ti),p_D2(ti)] = corr([kids_ages;adult_ages],prob_D2(:,ti));
[corrc_D5(ti),p_D5(ti)] = corr([kids_ages;adult_ages],prob_D5(:,ti));
end

figure('Name','mod vs age correlation over time')
subplot(121)
plot(trl_t,corrc_D2)
xlabel('t');ylabel('\rho prob vs age');hold on
title('D2')

plot(trl_t,((p_D2<0.05).*1)-1,'r-*')
yline(0)
subplot(122)
plot(trl_t,corrc_D5)
xlabel('t');ylabel('\rho prob vs age');hold on

plot(trl_t,((p_D5<0.05).*1)-1,'r-*')
yline(0)
title('D5')

PMBR_inds = trl_t >= 1 & trl_t <= 1.5;
MRBD_inds = trl_t >= 0.3 & trl_t <= 0.8;
get_mod = @(brob_tc) mean(brob_tc(:,PMBR_inds),2) - mean(brob_tc(:,MRBD_inds),2);
modsD2.ayoungest_hmm_burst_prob_D2_mod = get_mod(youngest_hmm_burst_prob_D2);
modsD5.ayoungest_hmm_burst_prob_D5_mod = get_mod(youngest_hmm_burst_prob_D5);
modsD2.bmiddle_hmm_burst_prob_D2_mod = get_mod(middle_hmm_burst_prob_D2);
modsD5.bmiddle_hmm_burst_prob_D5_mod = get_mod(middle_hmm_burst_prob_D5);
modsD2.coldest_hmm_burst_prob_D2_mod = get_mod(oldest_hmm_burst_prob_D2);
modsD5.coldest_hmm_burst_prob_D5_mod = get_mod(oldest_hmm_burst_prob_D5);
    
modsD2.dyoungest_hmm_burst_prob_D2_ad_mod = get_mod(youngest_hmm_burst_prob_D2_ad);
modsD5.dyoungest_hmm_burst_prob_D5_ad_mod = get_mod(youngest_hmm_burst_prob_D5_ad);
modsD2.emiddle_hmm_burst_prob_D2_ad_mod = get_mod(middle_hmm_burst_prob_D2_ad);
modsD5.emiddle_hmm_burst_prob_D5_ad_mod = get_mod(middle_hmm_burst_prob_D5_ad);
modsD2.foldest_hmm_burst_prob_D2_ad_mod = get_mod(oldest_hmm_burst_prob_D2_ad);
modsD5.foldest_hmm_burst_prob_D5_ad_mod = get_mod(oldest_hmm_burst_prob_D5_ad);

figure('Color','w','Position',[680 470 630 508], ...
    'Name','prob mod groups violin')

subplot(1,2,1)
vs_mod = violinplot(modsD2);
ylabel('burst prob. modulation');
xticklabels({'youngest','middle','oldest','adults_Y','adults_M','dadults_O'});
title('D2')

subplot(1,2,2)
vs_mod = violinplot(modsD5);
ylabel('burst prob. modulation');
xticklabels({'youngest','middle','oldest','adults_Y','adults_M','dadults_O'});
title('D5')
%% 

%% modulation vs age
mod_kids_D2 = get_mod(cell2mat(cellfun(cmean,hmm_results_kids.burst_raster_D1,'UniformOutput',false)));
mod_kids_D5 = get_mod(cell2mat(cellfun(cmean,hmm_results_kids.burst_raster_D4,'UniformOutput',false)));
mod_adults_D2 = get_mod(cell2mat(cellfun(cmean,hmm_results_adults.burst_raster_D1,'UniformOutput',false)));
mod_adults_D5 = get_mod(cell2mat(cellfun(cmean,hmm_results_adults.burst_raster_D4,'UniformOutput',false)));
lines_c = lines(2);


[rho_D5,p_D5] = corr([kids_ages;adult_ages],[mod_kids_D5;mod_adults_D5],'type','Pearson');
[rho_D2,p_D2] = corr([kids_ages;adult_ages],[mod_kids_D2;mod_adults_D2],'type','Pearson');


%%
f_burstmod_v_age = figure('Name','Beta Mod v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 10;
fheight = 6;
if badtrl
fwidth = 8;
fheight = 6;
end
f_burstmod_v_age.Position([3,4]) = [fwidth,fheight];
scatter(kids_ages,mod_kids_D2,'k^')
hold on
scatter(adult_ages,mod_adults_D2,'ko')
ylabel('Burst probability modulation (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [mod_kids_D2;mod_adults_D2];
b = X\y;
[R_burstmod_D2,p_burstmod_D2] = corr(x,y);
refl = refline(b(2),b(1));

set(f_burstmod_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig5/Burst_mod_D2.png',results_dir),'-dpng','-r900');
legend(refl,{sprintf('R^2 = %1.2f\np = %1.1e',R_burstmod_D2^2,p_burstmod_D2)}, ...
        'box','on','Location','northwest','NumColumns',1,'Color','w')
print(gcf,sprintf('%sFig5/Burst_mod_D2_W_legend.png',results_dir),'-dpng','-r900');

%%% burst prob mod vs beta power mod
burst_mods_Z = [zscore(mod_kids_D2);zscore(mod_adults_D2)];
beta_mods_Z = [zscore(TFS_results_kids.beta_modulation_D2);zscore(TFS_results_adults.beta_modulation_D2)];

f_burstmod_v_prob_mod = figure('Name','Beta Mod v age D2', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 10;
fheight = 5.5;
f_burstmod_v_prob_mod.Position([3,4]) = [fwidth,fheight];
scatter(zscore(mod_kids_D2),zscore(TFS_results_kids.beta_modulation_D2),'k^')
hold on
scatter(zscore(mod_adults_D2),zscore(TFS_results_adults.beta_modulation_D2),'ko')
xlabel('burst probability modulation (A.U.)')
ylabel('\beta-modulation (A.U.)')
set(lns,'HandleVisibility','off')
axis equal

x = burst_mods_Z;
X = [ones(length(x),1) x];
y = beta_mods_Z;
b = X\y;
[R_burstvbetamod_D2,p_burstvbetamod_D2] = corr(x,y);
refline(b(2),b(1));

set(f_burstmod_v_prob_mod.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig5/Burst_mod_v_power_mod_D2.png',results_dir),'-dpng','-r900');
lh=legend({'Children','Adults',sprintf('R^2 = %1.2f\np = %1.1e', ...
    R_burstvbetamod_D2^2,p_burstvbetamod_D2)}, ...
    'box','on','Location','southeast','NumColumns',1);
lh.Position = [0.6191,0.1833,0.3520,0.2996];
print(gcf,sprintf('%sFig5/Burst_mod_v_power_mod_D2_w_legend.png',results_dir),'-dpng','-r900');

%% %D5
f_burstmod_v_age = figure('Name','Beta Mod v age D5', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 10;
fheight = 6;
if badtrl
fwidth = 8;
fheight = 6;
end
f_burstmod_v_age.Position([3,4]) = [fwidth,fheight];
scatter(kids_ages,mod_kids_D5,'k^')
hold on
scatter(adult_ages,mod_adults_D5,'ko')
ylabel('Burst probability modulation (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [mod_kids_D5;mod_adults_D5];
b = X\y;
[R_burstmod_D5,p_burstmod_D5] = corr(x,y);
refl = refline(b(2),b(1));

set(f_burstmod_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig5/Burst_mod_D5.png',results_dir),'-dpng','-r900');
legend(refl,{sprintf('R^2 = %1.2f\np = %1.1e',R_burstmod_D5^2,p_burstmod_D5)}, ...
        'box','on','Location','northwest','NumColumns',1,'Color','w')
print(gcf,sprintf('%sFig5/Burst_mod_D5_w_legend.png',results_dir),'-dpng','-r900');

%%% burst prob mod vs beta power mod
burst_mods_Z = [zscore(mod_kids_D5);zscore(mod_adults_D5)];
beta_mods_Z = [zscore(TFS_results_kids.beta_modulation_D5);zscore(TFS_results_adults.beta_modulation_D5)];

f_burstmod_v_prob_mod = figure('Name','Beta Mod v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 10;
fheight = 5.5;
f_burstmod_v_prob_mod.Position([3,4]) = [fwidth,fheight];
scatter(zscore(mod_kids_D5),zscore(TFS_results_kids.beta_modulation_D5),'k^')
hold on
scatter(zscore(mod_adults_D5),zscore(TFS_results_adults.beta_modulation_D5),'ko')
xlabel('burst probability modulation (A.U.)')
ylabel('\beta-modulation (A.U.)')
set(lns,'HandleVisibility','off')
axis equal

x = burst_mods_Z;
X = [ones(length(x),1) x];
y = beta_mods_Z;
b = X\y;
[R_burstvbetamod_D5,p_burstvbetamod_D5] = corr(x,y);
refl = refline(b(2),b(1));

set(f_burstmod_v_prob_mod.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig5/Burst_mod_v_power_mod_D5.png',results_dir),'-dpng','-r900');
lh=legend({'Children','Adults',sprintf('R^2 = %1.2f\np = %1.1e', ...
    R_burstvbetamod_D5^2,p_burstvbetamod_D5)}, ...
    'box','on','Location','southeast','NumColumns',1);
lh.Position = [0.6191,0.1833,0.3520,0.2996];
print(gcf,sprintf('%sFig5/Burst_mod_v_power_mod_D5_w_legend.png',results_dir),'-dpng','-r900');

clear *TFS*

%%
%% D2 burst freq v age

f_burstfreq_v_age = figure('Name','Burst freq v age D2', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 10;
fheight = 6;
f_burstfreq_v_age.Position([3,4]) = [fwidth,fheight];
scatter(kids_ages,hmm_results_kids.burst_frequency_D1,'k^')
hold on
scatter(adult_ages,hmm_results_adults.burst_frequency_D1,'ko')
ylabel('Burst frequency (1/s)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [hmm_results_kids.burst_frequency_D1;hmm_results_adults.burst_frequency_D1];
b = X\y;
[R_burstf_D2,p_burstf_D2] = corr(x,y);
refl = refline(b(2),b(1));
legend(refl,{sprintf('R^2 = %1.2f\np = %1.1e',R_burstf_D2^2,p_burstf_D2)}, ...
        'box','on','Location','northeast','NumColumns',1,'Color','w')
set(f_burstfreq_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig6/Burst_freq_D2.png',results_dir),'-dpng','-r900');
%% D5 burst freq v age
f_burstfreq_v_age = figure('Name','Burst freq v age D5', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 10;
fheight = 6;
f_burstfreq_v_age.Position([3,4]) = [fwidth,fheight];
scatter(kids_ages,hmm_results_kids.burst_frequency_D4,'k^')
hold on
scatter(adult_ages,hmm_results_adults.burst_frequency_D4,'ko')
ylabel('Burst frequency (1/s)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [hmm_results_kids.burst_frequency_D4;hmm_results_adults.burst_frequency_D4];
b = X\y;
[R_burstf_D5,p_burstf_D5] = corr(x,y);
refl = refline(b(2),b(1));
legend(refl,{sprintf('R^2 = %1.2f\np = %1.1e',R_burstf_D5^2,p_burstf_D5)}, ...
        'box','on','Location','northeast','NumColumns',1,'Color','w')
set(f_burstfreq_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig6/Burst_freq_D5.png',results_dir),'-dpng','-r900');

%% %%%%%%%%%%%%%%%%%%%%%%%%
% Burst amplitude
f_burstamp_v_age = figure('Name','Burst amp v age D2', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 8.9;
fheight = 6;
f_burstamp_v_age.Position([3,4]) = [fwidth,fheight];
scatter(kids_ages,hmm_results_kids.burst_amplitude_D1,'k^')
hold on
scatter(adult_ages,hmm_results_adults.burst_amplitude_D1,'ko')
ylabel('Burst amplitude (A.U.)')
xlabel('Age (years)')
% lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
% lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
% lns(3) = xline(max(kids_ages)+1);
% set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [hmm_results_kids.burst_amplitude_D1;hmm_results_adults.burst_amplitude_D1];
b = X\y;
[R_burstamp_D2,p_burstamp_D2] = corr(x,y);
refl = refline(b(2),b(1));
legend(refl,{sprintf('R^2 = %1.2f\np = %1.3f',R_burstamp_D2^2,p_burstamp_D2)}, ...
        'box','on','Location','southwest','NumColumns',1,'Color','w')
set(f_burstamp_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig6/Burst_amp_D2_w_legend.png',results_dir),'-dpng','-r900');
legend off
set(f_burstamp_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig6/Burst_amp_D2.png',results_dir),'-dpng','-r900');

%%%%%%%%%%%
f_burstamp_v_age = figure('Name','Burst amp v age D5', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');

f_burstamp_v_age.Position([3,4]) = [fwidth,fheight];
scatter(kids_ages,hmm_results_kids.burst_amplitude_D4,'k^')
hold on
scatter(adult_ages,hmm_results_adults.burst_amplitude_D4,'ko')
ylabel('Burst amplitude (A.U.)')
xlabel('Age (years)')
% lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
% lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
% lns(3) = xline(max(kids_ages)+1);
% set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])

x = [kids_ages;adult_ages];
X = [ones(length(x),1) x];
y = [hmm_results_kids.burst_amplitude_D4;hmm_results_adults.burst_amplitude_D4];
b = X\y;
[R_burstamp_D5,p_burstamp_D5] = corr(x,y);
refl = refline(b(2),b(1));
legend(refl,{sprintf('R^2 = %1.2f\np = %1.3f',R_burstamp_D5^2,p_burstamp_D5)}, ...
        'box','on','Location','southwest','NumColumns',1,'Color','w')
set(f_burstamp_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig6/Burst_amp_D5_w_legend.png',results_dir),'-dpng','-r900');
legend off
set(f_burstamp_v_age.Children,'FontName','Arial','FontSize',9)
axx=gca;axx.Position(2)=0.16;
print(gcf,sprintf('%sFig6/Burst_amp_D5.png',results_dir),'-dpng','-r900');

%%
clear hmm_results*


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Beta connectivity
col_lim = 0.25;
pctl = 0.95;

youngest_AEC = reshape([AEC_results_kids.conn_mat{youngest}],[size(AEC_results_kids.conn_mat{1}),length(youngest)]);
middle_AEC = reshape([AEC_results_kids.conn_mat{middle}],[size(AEC_results_kids.conn_mat{1}),length(middle)]);
oldest_AEC = reshape([AEC_results_kids.conn_mat{oldest}],[size(AEC_results_kids.conn_mat{1}),length(oldest)]);

    
youngest_AEC_ad = reshape([AEC_results_adults.conn_mat{youngest_ad}],[size(AEC_results_adults.conn_mat{1}),length(youngest_ad)]);
middle_AEC_ad = reshape([AEC_results_adults.conn_mat{middle_ad}],[size(AEC_results_adults.conn_mat{1}),length(middle_ad)]);
oldest_AEC_ad = reshape([AEC_results_adults.conn_mat{oldest_ad}],[size(AEC_results_adults.conn_mat{1}),length(oldest_ad)]);
adults_AEC = reshape([AEC_results_adults.conn_mat{:}],[size(AEC_results_adults.conn_mat{1}),height(AEC_results_adults)]);

plot_mat_n_brains(mean(youngest_AEC,3),col_lim,pctl)
set(gcf,'Name','AEC youngest')
print(gcf,sprintf('%sFig4/YoungestAEC.png',results_dir),'-dpng','-r900');

plot_mat_n_brains(mean(middle_AEC,3),col_lim,pctl)
set(gcf,'Name','AEC middle')
print(gcf,sprintf('%sFig4/middleAEC.png',results_dir),'-dpng','-r900');

plot_mat_n_brains(mean(oldest_AEC,3),col_lim,pctl)
set(gcf,'Name','AEC oldest')
print(gcf,sprintf('%sFig4/oldestAEC.png',results_dir),'-dpng','-r900');

plot_mat_n_brains(mean(youngest_AEC_ad,3),col_lim,pctl)
set(gcf,'Name','AEC Adults 1')
print(gcf,sprintf('%sFig4/adults1AEC.png',results_dir),'-dpng','-r900');

plot_mat_n_brains(mean(middle_AEC_ad,3),col_lim,pctl)
set(gcf,'Name','AEC Adults 2')
print(gcf,sprintf('%sFig4/adults2AEC.png',results_dir),'-dpng','-r900');

plot_mat_n_brains(mean(oldest_AEC_ad,3),col_lim,pctl)
set(gcf,'Name','AEC Adults 3')
print(gcf,sprintf('%sFig4/adults3AEC.png',results_dir),'-dpng','-r900');

plot_mat_n_brains(mean(adults_AEC,3),col_lim,pctl)
set(gcf,'Name','AEC Adults all')
print(gcf,sprintf('%sFig4/adults_all_AEC.png',results_dir),'-dpng','-r900');
%%%%


%%%%%
%% strengts
conn_str=[];
conn_str.ayoungest = AEC_results_kids.conn_strength(youngest);
conn_str.bmiddle = AEC_results_kids.conn_strength(middle);
conn_str.coldest = AEC_results_kids.conn_strength(oldest);
    
conn_str.dadults_1=  AEC_results_adults.conn_strength(youngest_ad);
conn_str.dadults_2 = AEC_results_adults.conn_strength(middle_ad);
conn_str.dadults_3 = AEC_results_adults.conn_strength(oldest_ad);

% ylims = [-0.2,0.6];
figure('Color','w','Position',[680 470 630 508])
violinplot(conn_str);
ylabel('Conn. strenght');
xticklabels({'youngest','middle','oldest','adults_Y','adults_M','dadults_O'});

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_ages = [kids_ages;adult_ages];

all_conn_str = [AEC_results_kids.conn_strength;AEC_results_adults.conn_strength]./(77*78);
[conn_str_all_corr.Rho,conn_str_all_corr.p] = corr(all_ages,all_conn_str);

f_conn_v_age = figure('Name','Conn v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 8.5;
fheight = 5;
if badtrl
fwidth = 8;
fheight = 6;
end
f_conn_v_age.Position([3,4]) = [fwidth,fheight];
sc = scatter(all_ages,all_conn_str,20,'ko');
scatter(kids_ages,AEC_results_kids.conn_strength./(77*78),20,'k^')
hold on
scatter(adult_ages,AEC_results_adults.conn_strength./(77*78),20,'ko')

ylabel(' Global Connectivity (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])
% ylim([0,0.35])
x = all_ages;
X = [ones(length(x),1) x];
y = all_conn_str;
b = X\y;
refl = refline(b(2),b(1));

set(f_conn_v_age.Children,'FontName','Arial','FontSize',9)
print(f_conn_v_age,sprintf('%sFig4/Conn_v_age.png',results_dir),'-dpng','-r900');

legend(refl, {sprintf('R^2 = %1.3f\np = %1.2e',conn_str_all_corr.Rho.^2,conn_str_all_corr.p)}, ...
        'box','on','Location','northwest','NumColumns',1,'Color','w')
print(f_conn_v_age,sprintf('%sFig4/Conn_v_age_w_legend.png',results_dir),'-dpng','-r900');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('net_mats_mask.mat', 'net_mats_mask')
nets.lat_vis = net_mats_mask(:,:,1);
nets.central_vis = net_mats_mask(:,:,2);
nets.lat_vis2 = net_mats_mask(:,:,3);
% nets.bilat_parietal = net_mats_mask(:,:,4);
% nets.sens_mot = net_mats_mask(:,:,5);
% nets.med_parietal = net_mats_mask(:,:,6);
% nets.default_mod = net_mats_mask(:,:,7);
nets.left_frontopar = net_mats_mask(:,:,8);
nets.right_frontopar = net_mats_mask(:,:,9);
% nets.frontal = net_mats_mask(:,:,10);

figure
conn_str_central_vis=[];
conn_str_central_vis.ayoungest = AEC_results_kids.central_vis(youngest);
conn_str_central_vis.bmiddle = AEC_results_kids.central_vis(middle);
conn_str_central_vis.coldest = AEC_results_kids.central_vis(oldest);
    
conn_str_central_vis.dadults_1=  AEC_results_adults.central_vis(youngest_ad);
conn_str_central_vis.dadults_2 = AEC_results_adults.central_vis(middle_ad);
conn_str_central_vis.dadults_3 = AEC_results_adults.central_vis(oldest_ad);
violinplot(conn_str_central_vis);
% xticklabels({ayoungest,})
title('central visual')

fgb = figure('Name','Central vis brain','Units','centimeters');
fgb.Position([3,4]) = 5.*[3,2];
net_ = (nets.central_vis).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
set(gca,'Position',[0,0,1,1])
print(gcf,sprintf('%sFig4/central_vis.png',results_dir),'-dpng','-r600');

figure
conn_str_lat_vis=[];
conn_str_lat_vis.ayoungest = AEC_results_kids.lat_vis(youngest);
conn_str_lat_vis.bmiddle = AEC_results_kids.lat_vis(middle);
conn_str_lat_vis.coldest = AEC_results_kids.lat_vis(oldest);
    
conn_str_lat_vis.dadults_1=  AEC_results_adults.lat_vis(youngest_ad);
conn_str_lat_vis.dadults_2 = AEC_results_adults.lat_vis(middle_ad);
conn_str_lat_vis.dadults_3 = AEC_results_adults.lat_vis(oldest_ad);
violinplot(conn_str_lat_vis);
title('lateral visual')

fgb = figure('Name','lateral vis brain','Units','centimeters');
fgb.Position([3,4]) = 5.*[3,2];
net_ = (nets.lat_vis).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
set(gca,'Position',[0,0,1,1])
print(gcf,sprintf('%sFig4/lateral_vis.png',results_dir),'-dpng','-r600');
%%
figure
conn_str_left_FP=[];
conn_str_left_FP.ayoungest = AEC_results_kids.left_frontopar(youngest);
conn_str_left_FP.bmiddle = AEC_results_kids.left_frontopar(middle);
conn_str_left_FP.coldest = AEC_results_kids.left_frontopar(oldest);
    
conn_str_left_FP.dadults_1=  AEC_results_adults.left_frontopar(youngest_ad);
conn_str_left_FP.dadults_2 = AEC_results_adults.left_frontopar(middle_ad);
conn_str_left_FP.dadults_3 = AEC_results_adults.left_frontopar(oldest_ad);
violinplot(conn_str_left_FP);
title('left frontoparietal')

fgb = figure('Name','left FP brain','Units','centimeters');
fgb.Position([3,4]) = 5.*[3,2];
net_ = (nets.left_frontopar).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
set(gca,'Position',[0,0,1,1])
print(gcf,sprintf('%sFig4/left_FP.png',results_dir),'-dpng','-r900');
%%
figure
conn_str_right_FP=[];
conn_str_right_FP.ayoungest = AEC_results_kids.right_frontopar(youngest);
conn_str_right_FP.bmiddle = AEC_results_kids.right_frontopar(middle);
conn_str_right_FP.coldest = AEC_results_kids.right_frontopar(oldest);
    
conn_str_right_FP.dadults_1=  AEC_results_adults.right_frontopar(youngest_ad);
conn_str_right_FP.dadults_2 = AEC_results_adults.right_frontopar(middle_ad);
conn_str_right_FP.dadults_3 = AEC_results_adults.right_frontopar(oldest_ad);
vs = violinplot(conn_str_right_FP);
title('right frontoparietal')

fgb = figure('Name','right FP brain','Units','centimeters');
fgb.Position([3,4]) = 5.*[3,2];
net_ = (nets.right_frontopar).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
set(gca,'Position',[0,0,1,1])
print(gcf,sprintf('%sFig4/right_FP.png',results_dir),'-dpng','-r900');
%%
subnet_conn_all.lat_vis = [AEC_results_kids.lat_vis;AEC_results_adults.lat_vis]./(sum(nets.lat_vis(:))./2);
subnet_conn_all.central_vis =  [AEC_results_kids.central_vis;AEC_results_adults.central_vis]./(sum(nets.central_vis(:))./2);
subnet_conn_all.right_frontopar = [AEC_results_kids.right_frontopar;AEC_results_adults.right_frontopar]./(sum(nets.right_frontopar(:))./2);
subnet_conn_all.left_frontopar =  [AEC_results_kids.left_frontopar;AEC_results_adults.left_frontopar]./(sum(nets.left_frontopar(:))./2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cols = [166,206,227
31,120,180
178,223,138
51,160,44]./255;
all_ages = [kids_ages;adult_ages];

[conn_str_lat_vis.Rho,conn_str_lat_vis.p] = corr(all_ages,subnet_conn_all.lat_vis);
[central_vis.Rho,central_vis.p] = corr(all_ages,subnet_conn_all.central_vis);
[conn_str_left_FP.Rho,conn_str_left_FP.p] = corr(all_ages,subnet_conn_all.left_frontopar);
[conn_str_right_FP.Rho,conn_str_right_FP.p] = corr(all_ages,subnet_conn_all.right_frontopar);
ylabel('beta global Connectivity strength');xlabel('Age')

f_conn_v_age_vis = figure('Name','visual Conn v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 7.8;
fheight = 5;
f_conn_v_age_vis.Position([3,4]) = [fwidth,fheight];
scatter(all_ages,subnet_conn_all.lat_vis,20,'ko','MarkerFaceColor',cols(1,:))
hold on
scatter(all_ages,subnet_conn_all.central_vis,20,'k^','MarkerFaceColor',cols(3,:))
ylabel('Connectivity (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])
ylim([0,0.8])
x = all_ages;
X = [ones(length(x),1) x];
y = subnet_conn_all.lat_vis;
b = X\y;
refl1 = refline(b(2),b(1));refl1.Color = cols(1,:);   
y = subnet_conn_all.central_vis;
b = X\y;
refl2 = refline(b(2),b(1));refl2.Color = cols(3,:); 
legend([refl1,refl2], {sprintf('R^2 = %1.3f, p = %1.2e',conn_str_lat_vis.Rho.^2,conn_str_lat_vis.p), ...
        sprintf('R^2 = %1.3f, p = %1.2e',central_vis.Rho.^2,central_vis.p)},...
        'box','on','Location','best','NumColumns',1,'Color','w')
set(f_conn_v_age_vis.Children,'FontName','Arial','FontSize',9)
print(f_conn_v_age_vis,sprintf('%sFig4/VisualConn_v_age.png',results_dir),'-dpng','-r900');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_conn_v_age_frontopar = figure('Name','frontopar Conn v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 7.8;
fheight = 5;
f_conn_v_age_frontopar.Position([3,4]) = [fwidth,fheight];
scatter(all_ages,subnet_conn_all.left_frontopar,20,'ko','MarkerFaceColor',cols(2,:))
hold on
scatter(all_ages,subnet_conn_all.right_frontopar,20,'k^','MarkerFaceColor',cols(4,:))
ylabel('Connectivity (A.U.)')
xlabel('Age (years)')
lns(1) = xline((max(kids_ages(youngest)) + min(kids_ages(middle)))./2);
lns(2) = xline((max(kids_ages(middle)) + min(kids_ages(oldest)))./2);
lns(3) = xline(max(kids_ages)+1);
set(lns,'HandleVisibility','off')
xlim([0,1+max(adult_ages)])
ylim([0,0.8])
x = all_ages;
X = [ones(length(x),1) x];
y = subnet_conn_all.left_frontopar;
b = X\y;
refl1 = refline(b(2),b(1));refl1.Color = cols(2,:);   
y = subnet_conn_all.right_frontopar;
b = X\y;
refl2 = refline(b(2),b(1));refl2.Color = cols(4,:); 
legend([refl1,refl2], {sprintf('R^2 = %1.3f, p = %1.2e',conn_str_left_FP.Rho.^2,conn_str_left_FP.p), ...
        sprintf('R^2 = %1.3f, p = %1.2e',conn_str_right_FP.Rho.^2,conn_str_right_FP.p)},...
        'box','on','Location','best','NumColumns',1,'Color','w')
set(f_conn_v_age_frontopar.Children,'FontName','Arial','FontSize',9)
print(f_conn_v_age_frontopar,sprintf('%sFig4/FrontoparConn_v_age.png',results_dir),'-dpng','-r900');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Region strength v age
addpath('.\BrainPlots\')
addpath('.\gifti-1.8\')
conn_degree_all = [AEC_results_kids.conn_degree;AEC_results_adults.conn_degree];
f_regionconn_v_age = figure('Name','Region-Conn v age', ...
    'Units','centimeters','Color','w', ...
    'Renderer','painters');
fwidth = 9.5;
fheight = 5.7;
f_regionconn_v_age.Position([3,4]) = [fwidth,fheight];

x = all_ages;
X = [ones(length(x),1) x];
fit_grad = nan(size(all_ages));
for reg = 1:78
    y = conn_degree_all(:,reg);
    b = X\y;
    fit_grad(reg) = b(2);
    [r_reg(reg),p_reg(reg)] = corr(all_ages,y);
end  
ax1 = axes; %top
PaintBrodmannAreas_chooseview(fit_grad, ...
    78, 256, [], [], [], [])
ax1.Position = [0.25,0.14,0.5,0.57];
axis equal

ax2 = axes; %left
PaintBrodmannAreas_chooseview(fit_grad, ...
    78, 256, [], [], [], [-90,0])
ax2.Position = [-0.05,0.5,0.5,0.5];
axis equal

ax3 = axes; %right
PaintBrodmannAreas_chooseview(fit_grad, ...
    78, 256, [], [], [], [90,0])
ax3.Position = [0.55,0.5,0.5,0.5];
axis equal

ax4 = axes; %back
PaintBrodmannAreas_chooseview(fit_grad, ...
    78, 256, [], [], [], [0,0])
ax4.Position = [0.55,0.07,0.5,0.5];
axis equal

ax5 = axes; %front
PaintBrodmannAreas_chooseview(fit_grad, ...
    78, 256, [], [], [], [-180,0])
ax5.Position = [-0.05,0.07,0.5,0.5];
axis equal
cb = colorbar('Location','South','Position',[0.2,0.1,0.6,0.02]);

set(f_regionconn_v_age.Children,'FontName','Arial','FontSize',9)
print(f_regionconn_v_age,sprintf('%sFig4/RegionConn_v_age.png',results_dir),'-dpng','-r900');

%% example scatters
example_regs = [8,8+39,23,23+39];
f_fits = fopen(sprintf('%sFig4/RegionConn_v_age_fitlines.txt',results_dir),'w');
for reg_i = 1:4
    reg = example_regs(reg_i);
    f_regionconn_v_age_example = figure('Name','Region-Conn v age example', ...
        'Units','centimeters',...
        'Color','w', ...
        'Renderer','painters');
    fwidth = 4.45;
    fheight = 5.945;
    f_regionconn_v_age_example.Position([3,4]) = [fwidth,fheight];
    ax = axes;
    cols = [217,95,2
        117,112,179]./255;
    scatter(kids_ages,AEC_results_kids.conn_degree(:,reg),10,'k^');hold on
    scatter(adult_ages,AEC_results_adults.conn_degree(:,reg),10,'ko')

    ylim([0,20])
    x = all_ages;
    X = [ones(length(x),1) x];
    y = conn_degree_all(:,reg);
    b = X\y;
    refl1 = refline(b(2),b(1));
    [R_conn_reg,p_conn_reg] = corr(x,y);
    fprintf(f_fits,['Example Region %d, (AAL %d), R^2=%1.2f, p=%1.1e\n',...
        'Conn. Degree = %1.3f*age + %1.3f\n\n'],...
        reg_i,reg,R_conn_reg.^2,p_conn_reg,...
        b(2),b(1));
    fprintf('Example Region %d, (AAL %d), R^2=%1.2f, p=%1.1e\n',reg_i,reg,R_conn_reg.^2,p_conn_reg);
    ylabel('Degree(A.U.)')
    xlabel('Age (years)')
    ax.Position = [0.2 0.17 0.7246 0.66];

    ax2 = axes;
    sig_regs = nan(78,1);sig_regs(reg)=70;
    % PaintBrodmannAreas_chooseview(1:78, 78, 256, [0,79], [], [], []);
    if reg_i <3
        PaintBrodmannAreas_chooseview(sig_regs, 78, 256, [0,79], [], [], [])
        ax2.Position = [0.1,0.65,0.6,0.3];
    else
        PaintBrodmannAreas_chooseview(sig_regs, ...
            78, 256, [0,79], [], [], [0,0])
        ax2.Position = [0.2,0.65,0.6,0.3];
    end

    set(f_regionconn_v_age_example.Children,'FontName','Arial','FontSize',9)
    print(f_regionconn_v_age_example,sprintf('%sFig4/RegionConn_v_ageexample%d.png',results_dir,reg_i),'-dpng','-r900');
end
fclose(f_fits);
%%
figure
sig_regs = nan(78,1);sig_regs(8+39)=1;
% PaintBrodmannAreas_chooseview(1:78, 78, 256, [0,79], [], [], []);
PaintBrodmannAreas_chooseview(sig_regs, 78, 256, [0,79], [], [], [])

figure
sig_regs = nan(78,1);sig_regs(23)=1;
% PaintBrodmannAreas_chooseview(1:78, 78, 256, [0,79], [], [], []);
PaintBrodmannAreas_chooseview(sig_regs, 78, 256, [0,79], [], [], [])
%% % -----------------------------------------------------------------------
%%% subfunctions
%%% -----------------------------------------------------------------------
function plot_TFS(TFS,cbar_lim,mean_evoked,std_evoked)
fre = 3:2:50;

time = linspace(0,size(TFS,2)./1200,size(TFS,2));
pcolor(time,fre,TFS);shading interp
xlabel('Time (s)');
ylabel('Frequency (Hz)')
%cbar_lim = 0.5;
cb = colorbar('Location','southoutside'); caxis([-cbar_lim cbar_lim])
cb.Label.String = 'Fractional change';
axis fill
ylim([3,50])

hold on

evoked_offset = 40;
evoked_scale = 1/100;
mean_evoked_sc = (mean_evoked.*evoked_scale) + evoked_offset;
std_evoked_sc = std_evoked.*evoked_scale;
plot(time,mean_evoked_sc,'k')
ciplot(mean_evoked_sc-std_evoked_sc,...
    mean_evoked_sc+std_evoked_sc,time,'k')
yline(evoked_offset,'k')
xline(0.132,'k','LineWidth',0.5)

yyaxis right
ax2=gca;
ax2.YColor = 'k';
ylim([(fre(1)-fre(end)+fre(end)-evoked_offset)/evoked_scale,10/evoked_scale])
yticks([-10/evoked_scale,0,10/evoked_scale])
% yticklabels({sprintf("%1.0e",-),0,sprintf("%1.0e",10/evoked_scale)})

yline(-10/evoked_scale,'--k','LineWidth',0.01)
ax2.YAxis(2).Exponent=3;
yl=ylabel('Amp. (A.U.)');%yl.Position(2)=0;
end

function plot_mat_n_brains(AECmat,col_lim,pctl)
fh = figure;
set(fh,'Units','centimeters','Color','w','Renderer','painters');
fwidth = 3.8;
fheight = 5.9;
fh.Position([3,4]) = [fwidth,fheight];
ax_mat = axes;
imagesc(AECmat);cb = colorbar('Location','southoutside','FontSize',8);
if isempty(col_lim);col_lim = max(abs(AECmat(:)));end
cLims = [-1,1]*col_lim;
caxis(cLims);
axis square; 
yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
% xtickangle(45)
ax_mat.Position = [0.46,0.57,0.52,0.52];
ax_mat.FontSize=6;
% = [0.15,0.67];

drawnow
ax3 = axes;
ax3.Position = [-0.1,0.0,0.6,0.3];
go_netviewer_perctl_lim(AECmat,pctl,col_lim)
view([-180,0]) %FRONT VIEW
ax2 = axes;
ax2.Position = [0.4,0.0,0.6,0.4];
go_netviewer_perctl_lim(AECmat,pctl,col_lim)
view([-90,0]) %side view
ax1 = axes;ax1.Position = [-0.05,0.25,0.55,0.4];
go_netviewer_perctl_lim(AECmat,pctl,col_lim) %top view
% st = sgtitle(band_name{f_ind});
drawnow
set(fh,'Renderer','painters')

end

function plot_raster(raster_cell,grpsz)
imagesc(cell2mat(raster_cell))
hold on
colormap('gray')
ylabel('Trial Number')
xlabel('time (s)')
xlim([-0.1,3.8].*100)
box off
xticklabels(strsplit(num2str(xticks./100)))
trl_count = 0;
if ~isempty(grpsz)
    grpsz = cumsum(grpsz);
else
    grpsz=0;
end

for s = 1:length(raster_cell)
    trl_count = trl_count + size(raster_cell{s},1);
    lh(s) = plot([3.5,3.8].*100,[1,1].*trl_count+0.5,'r','LineWidth',1.5);
    if any(s==grpsz)
        lh(s).Color = 'b';
    end
end
end