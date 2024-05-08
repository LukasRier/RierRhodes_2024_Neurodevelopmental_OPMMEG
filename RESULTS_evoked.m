clearvars
close all
badtrls = 0
if badtrls
    project_dir =  '.\Adults_badtrls\';
else
    project_dir =  '.\Adults\';

end
EVOKED_results_adults = table({},...
    'VariableNames',{'ID'});
EVOKED_results_kids = table({},...
    'VariableNames',{'ID'});
conwin = [2.5,3];
extension_T = '';
load([project_dir,'.\sub_info_adults.mat'],'AdultDatasets')
AdultDatasets.Age = years(AdultDatasets.Date - AdultDatasets.DOB);
adults_ages = [];
%% evoked responses
% corr_inds =eval('1:end');

evoked_tc_index_all = [];
evoked_tc_pinky_all = [];
% good_subs = 1:26;good_subs(good_subs==20|good_subs==23)=[];
good_subs_adults = [1:19,21:22,24:26];
% good_subs_adults = 1:26
for sub_i = good_subs_adults
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
    EVOKED_results_adults.ID(height(EVOKED_results_adults)+1) = {sub};
    
    
    adults_ages = cat(1,adults_ages,AdultDatasets.Age(startsWith(AdultDatasets.SubjID,sub)));

    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';
    hp = 4;
    lp = 40;
    
    datadir = [project_dir,'Data',filesep,'BIDS',filesep];

    path_evoked = [datadir,'derivatives',filesep,'evoked',extension_T,filesep,'sub-',sub,filesep];

    %% envo
    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];
    files_peak_evoked = [filename,'_peak_evoked.mat'];
    load([path_evoked,files_peak_evoked],'index_evoked','little_evoked')

    evoked_tc_index_all = cat(1,evoked_tc_index_all,index_evoked');
    evoked_tc_pinky_all = cat(1,evoked_tc_pinky_all,little_evoked');
end

grp_mean_evoked_index_all = mean(evoked_tc_index_all,1);
grp_mean_evoked_pinky_all = mean(evoked_tc_pinky_all,1);

grp_std_evoked_index_all = std(evoked_tc_index_all,[],1);
grp_std_evoked_pinky_all = std(evoked_tc_pinky_all,[],1);
fs=1200;
control_window = round(conwin.*fs);control_inds = control_window(1):control_window(2);


%% index and pinky peak
figure
FtSz=14;
set(gcf,'Color','w')
set(gcf,'Position',[681 559 1120 420])

fs = 1200;
time = linspace(0,size(grp_mean_evoked_index_all,2)./fs,size(grp_mean_evoked_index_all,2));
hold on

plot(time,grp_mean_evoked_index_all,'b','LineWidth',2,...
    'Displayname','D1 trials desync peak');
ciplot(grp_mean_evoked_index_all-grp_std_evoked_index_all,...
    grp_mean_evoked_index_all+grp_std_evoked_index_all,time,'b')


lh = legend;lh.String{2} = 'std. dev.';
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylims=ylim;
% ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Evoked Amplitude')

% % pinky
figure
FtSz=14;
set(gcf,'Color','w')
set(gcf,'Position',[681 559 1120 420])

fs = 1200;
time = linspace(0,size(grp_mean_evoked_pinky_all,2)./fs,size(grp_mean_evoked_pinky_all,2));
hold on

plot(time,grp_mean_evoked_pinky_all,'b','LineWidth',2,...
    'Displayname','D5 trials desync peak');
ciplot(grp_mean_evoked_pinky_all-grp_std_evoked_pinky_all,...
    grp_mean_evoked_pinky_all+grp_std_evoked_pinky_all,time,'b')


lh = legend;lh.String{2} = 'std. dev.';
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylims=ylim;
% ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Evoked Amplitude')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kids
project_dir =  '.\Children\';

extension_T = '';
load(['sub_info.mat'],'KidsDatasets')
KidsDatasets.Age_yrs = years(KidsDatasets.Date - KidsDatasets.DOB);
kids_ages = [];
%% envelopes

evoked_tc_index_all_kids = [];
evoked_tc_pinky_all_kids = [];

good_subs = [1:27];
for sub_i = good_subs
    sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0'
    EVOKED_results_kids.ID(height(EVOKED_results_kids)+1) = {sub};
    kids_ages = cat(1,kids_ages,KidsDatasets.Age(startsWith(KidsDatasets.SubjID,sub)));
    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';
    hp = 4;
    lp = 40;

    datadir = [project_dir,'Data',filesep,'BIDS',filesep];

    path_evoked = [datadir,'derivatives',filesep,'evoked',extension_T,filesep,'sub-',sub,filesep];

    %% evoked data

    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];
    files_peak_evoked = [filename,'_peak_evoked.mat'];
    load([path_evoked,files_peak_evoked],'index_evoked','little_evoked')

    evoked_tc_index_all_kids = cat(1,evoked_tc_index_all_kids,index_evoked');
    evoked_tc_pinky_all_kids = cat(1,evoked_tc_pinky_all_kids,little_evoked');
end
[~,age_rank] = sort(kids_ages);

youngest = age_rank(1:floor(length(age_rank)/3));
oldest = age_rank(end - floor(length(age_rank)/3)+1 : end);
middle = age_rank(floor(length(age_rank)/3)+1:end - floor(length(age_rank)/3));

grp_mean_evoked_index_all_kids = mean(evoked_tc_index_all_kids,1);
grp_mean_evoked_pinky_all_kids = mean(evoked_tc_pinky_all_kids,1);

grp_std_evoked_index_all_kids = std(evoked_tc_index_all_kids,[],1);
grp_std_evoked_pinky_all_kids = std(evoked_tc_pinky_all_kids,[],1);


%% index and pinky desync peak kids
figure
FtSz=14;
set(gcf,'Color','w')
set(gcf,'Position',[681 559 1120 420])

fs = 1200;
time = linspace(0,size(grp_mean_evoked_index_all_kids,2)./fs,size(grp_mean_evoked_index_all_kids,2));
hold on

plot(time,grp_mean_evoked_index_all_kids,'m','LineWidth',2,...
    'Displayname','D1 trials desync peak kids');
ciplot(grp_mean_evoked_index_all_kids-grp_std_evoked_index_all_kids,...
    grp_mean_evoked_index_all_kids+grp_std_evoked_index_all_kids,time,'m')


lh = legend;lh.String{2} = 'std. dev.';
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylims=ylim;
% ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Evoked Amplitude')

% % pinky
figure
FtSz=14;
set(gcf,'Color','w')
set(gcf,'Position',[681 559 1120 420])

fs = 1200;
time = linspace(0,size(grp_mean_evoked_pinky_all_kids,2)./fs,size(grp_mean_evoked_pinky_all_kids,2));
hold on

plot(time,grp_mean_evoked_pinky_all_kids,'m','LineWidth',2,...
    'Displayname','D5 trials desync peak kids');
ciplot(grp_mean_evoked_pinky_all_kids-grp_std_evoked_pinky_all_kids,...
    grp_mean_evoked_pinky_all_kids+grp_std_evoked_pinky_all_kids,time,'m')


lh = legend;lh.String{2} = 'std. dev.';
chldrn = get(gca,'Children');chldrn(1).FaceAlpha=.3;
plot(xlim,[0,0],'k','HandleVisibility','off')
ylims=ylim;
% ylim(ylims)
ax= gca;
ax.FontSize=FtSz;
xlabel('Time(s)')
ylabel('Evoked Amplitude')

%% Flip using P50 and subgroups
t_ind_p50 = round(0.187545*fs);
t_ind_p50 = round(0.185*fs);

for sub_i = 1:size(evoked_tc_index_all,1)
    if evoked_tc_index_all(sub_i,t_ind_p50) < 0
        evoked_tc_index_all(sub_i,:) = -1.* evoked_tc_index_all(sub_i,:);
    end

    if evoked_tc_pinky_all(sub_i,t_ind_p50) < 0
        evoked_tc_pinky_all(sub_i,:) = -1.* evoked_tc_pinky_all(sub_i,:);
    end
end

for sub_i = 1:size(evoked_tc_index_all_kids,1)
    if evoked_tc_index_all_kids(sub_i,t_ind_p50) < 0
        evoked_tc_index_all_kids(sub_i,:) = -1.* evoked_tc_index_all_kids(sub_i,:);
    end
    
    if evoked_tc_pinky_all_kids(sub_i,t_ind_p50) < 0
        evoked_tc_pinky_all_kids(sub_i,:) = -1.* evoked_tc_pinky_all_kids(sub_i,:);
    end
end


%%%%% index kids
grp_mean_evoked_youngest_index = mean(evoked_tc_index_all_kids(youngest,:),1);
grp_std_evoked_youngest_index = std(evoked_tc_index_all_kids(youngest,:),1);

grp_mean_evoked_middle_index = mean(evoked_tc_index_all_kids(middle,:),1);
grp_std_evoked_middle_index = std(evoked_tc_index_all_kids(middle,:),1);

grp_mean_evoked_oldest_index = mean(evoked_tc_index_all_kids(oldest,:),1);
grp_std_evoked_oldest_index = std(evoked_tc_index_all_kids(oldest,:),1);
%%%%%% pinky kids
grp_mean_evoked_youngest_pinky = mean(evoked_tc_pinky_all_kids(youngest,:),1);
grp_std_evoked_youngest_pinky = std(evoked_tc_pinky_all_kids(youngest,:),1);

grp_mean_evoked_middle_pinky = mean(evoked_tc_pinky_all_kids(middle,:),1);
grp_std_evoked_middle_pinky = std(evoked_tc_pinky_all_kids(middle,:),1);

grp_mean_evoked_oldest_pinky = mean(evoked_tc_pinky_all_kids(oldest,:),1);
grp_std_evoked_oldest_pinky = std(evoked_tc_pinky_all_kids(oldest,:),1);

%adults

grp_mean_evoked_index = mean(evoked_tc_index_all,1);
grp_std_evoked_index = std(evoked_tc_index_all,1);

grp_mean_evoked_pinky = mean(evoked_tc_pinky_all,1);
grp_std_evoked_pinky = std(evoked_tc_pinky_all,1);

cols = lines(4);


%% index
figure
sgtitle('Index')
subplot(4,1,1)
plot(time,grp_mean_evoked_youngest_index,'Color',cols(1,:),'LineWidth',2,...
    'Displayname','Youngest');hold on
ciplot(grp_mean_evoked_youngest_index-grp_std_evoked_youngest_index,...
    grp_mean_evoked_youngest_index+grp_std_evoked_youngest_index,time,cols(1,:))
subplot(4,1,2)

plot(time,grp_mean_evoked_middle_index,'Color',cols(2,:),'LineWidth',2,...
    'Displayname','Middle');hold on
ciplot(grp_mean_evoked_middle_index-grp_std_evoked_middle_index,...
    grp_mean_evoked_middle_index+grp_std_evoked_middle_index,time,cols(2,:))
subplot(4,1,3)

plot(time,grp_mean_evoked_oldest_index,'Color',cols(3,:),'LineWidth',2,...
    'Displayname','Oldest');hold on
ciplot(grp_mean_evoked_oldest_index-grp_std_evoked_oldest_index,...
    grp_mean_evoked_oldest_index+grp_std_evoked_oldest_index,time,cols(3,:))
subplot(4,1,4)

plot(time,grp_mean_evoked_index,'Color',cols(4,:),'LineWidth',2,...
    'Displayname','Adults');hold on
ciplot(grp_mean_evoked_index - grp_std_evoked_index,...
    grp_mean_evoked_index+grp_std_evoked_index,time,cols(4,:))

lh = legend;
lh.String{2} = 'std. dev.';
lh.String{4} = 'std. dev.';
lh.String{6} = 'std. dev.';
lh.String{8} = 'std. dev.';
% chldrn = get(gca,'Children');set(chldrn([1,3,5,7]),'FaceAlpha',0);
yline(0,'k')
xline(0.187545,'LineWidth',0.5)

%% little
figure
sgtitle('Little')
subplot(4,1,1)
plot(time,grp_mean_evoked_youngest_pinky,'Color',cols(1,:),'LineWidth',2,...
    'Displayname','Youngest');hold on
ciplot(grp_mean_evoked_youngest_pinky-grp_std_evoked_youngest_pinky,...
    grp_mean_evoked_youngest_pinky+grp_std_evoked_youngest_pinky,time,cols(1,:))
subplot(4,1,2)

plot(time,grp_mean_evoked_middle_pinky,'Color',cols(2,:),'LineWidth',2,...
    'Displayname','Middle');hold on
ciplot(grp_mean_evoked_middle_pinky-grp_std_evoked_middle_pinky,...
    grp_mean_evoked_middle_pinky+grp_std_evoked_middle_pinky,time,cols(2,:))
subplot(4,1,3)

plot(time,grp_mean_evoked_oldest_pinky,'Color',cols(3,:),'LineWidth',2,...
    'Displayname','Oldest');hold on
ciplot(grp_mean_evoked_oldest_pinky-grp_std_evoked_oldest_pinky,...
    grp_mean_evoked_oldest_pinky+grp_std_evoked_oldest_pinky,time,cols(3,:))
subplot(4,1,4)

plot(time,grp_mean_evoked_pinky,'Color',cols(4,:),'LineWidth',2,...
    'Displayname','Adults');hold on
ciplot(grp_mean_evoked_pinky - grp_std_evoked_pinky,...
    grp_mean_evoked_pinky+grp_std_evoked_pinky,time,cols(4,:))

lh = legend;
lh.String{2} = 'std. dev.';
lh.String{4} = 'std. dev.';
lh.String{6} = 'std. dev.';
lh.String{8} = 'std. dev.';
% chldrn = get(gca,'Children');set(chldrn([1,3,5,7]),'FaceAlpha',0);
yline(0,'k')
xline(0.187545,'LineWidth',0.5)
%%
figure('Name','P50')
subplot(121)
amplitude = ([evoked_tc_index_all_kids(:,t_ind_p50);evoked_tc_index_all(:,t_ind_p50)]);
all_ages = [kids_ages;adults_ages];
X = [ones(length(all_ages),1) all_ages];
b = X\amplitude;
amplitude_model = X*b;
scatter(all_ages,amplitude,'ko')
hold on
plot(all_ages,amplitude_model,'r-')
[rho,pval] = corr(all_ages,amplitude);
xlabel("Age (years)")
title(sprintf("\\rho = %1.3f, p = %1.7f\n",rho,pval))
axis square
set(gcf,'Color','w')
ylabel('P50 amplitude (A.U.) Index')

subplot(122)
amplitude = ([evoked_tc_pinky_all_kids(:,t_ind_p50);evoked_tc_pinky_all(:,t_ind_p50)]);
b = X\amplitude;
amplitude_model = X*b;
scatter(all_ages,amplitude,'ko')
hold on
plot(all_ages,amplitude_model,'r-')
[rho,pval] = corr(all_ages,amplitude);
xlabel("Age (years)")
title(sprintf("\\rho = %1.3f, p = %1.7f\n",rho,pval))
axis square
set(gcf,'Color','w')
ylabel('P50 amplitude (A.U.) Little')



EVOKED_results_adults.evoked_D2 = evoked_tc_index_all;
EVOKED_results_kids.evoked_D2 = evoked_tc_index_all_kids;
EVOKED_results_adults.evoked_D5 = evoked_tc_pinky_all;
EVOKED_results_kids.evoked_D5 = evoked_tc_pinky_all_kids;
if badtrls
    save('Evoked_traces_badtrl.mat',"grp_mean_evoked*index","grp_mean_evoked*pinky","grp_std_evoked*index","grp_std_evoked*pinky")
    save('EVOKED_RESULTS_badtrl.mat','EVOKED_results_adults','EVOKED_results_kids','t_ind_p50')
else
    save('Evoked_traces.mat',"grp_mean_evoked*index","grp_mean_evoked*pinky","grp_std_evoked*index","grp_std_evoked*pinky")
    save('EVOKED_RESULTS.mat','EVOKED_results_adults','EVOKED_results_kids','t_ind_p50')
end
