clearvars
close all

AEC_results_adults = table({},...
    'VariableNames',{'ID'});
AEC_results_kids = table({},...
    'VariableNames',{'ID'});

if badtrls
    project_dir =  '.\Adults_badtrls\';
else
    project_dir =  '.\Adults\';

end
% conwin = [3,3.5];
% extension_T = '_3-3.5bl';
hp = 13;lp = 30;
load(['sub_info_adults.mat'],'AdultDatasets')
extension_T = '';
extension_AEC = '';
AEC_all = [];
AEC_all_norm = [];
% good_subs = [1:15,17:19,21:23];
good_subs = [1:19,21:22,24:26];
% good_subs = 1:26
f_conn_all = figure;f_conn_all.Color = 'w';set(f_conn_all,"Units","normalized","Position",[0,0,1,1])
TL = tiledlayout('flow','TileSpacing','Compact');
f_conn_all_mats = figure;f_conn_all_mats.Color = 'w';set(f_conn_all_mats,"Units","normalized","Position",[0,0,1,1])
TL2 = tiledlayout('flow','TileSpacing','Compact');
% lims = 0.2[]
adults_ages = [];
for sub_i = good_subs
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0'
        AEC_results_adults.ID(height(AEC_results_adults)+1) = {sub};

    adults_ages = cat(1,adults_ages,AdultDatasets.Age(startsWith(AdultDatasets.SubjID,sub)))
    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';

    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];

    datadir = [project_dir,'Data',filesep,'BIDS',filesep];

    path_AEC = [datadir,'derivatives',filesep,'AEC',extension_AEC,filesep,'sub-',sub,filesep];
    path_Tstat = [datadir,'derivatives',filesep,'Tstats',extension_T,filesep,'sub-',sub,filesep];

    files_AEC = [filename,'_AEC'];

    load(sprintf('%s%s_%d_%d_Hz_Z.mat',path_AEC,files_AEC,hp,lp))

    %     figure()
    %     subplot(121)
    %     imagesc(AEC);colorbar;
    %     axis square
    %     title(['Sub ',sub])
    %     subplot(122)
    figure(f_conn_all)
    nexttile
    title(sub)
    go_netviewer_perctl(AEC,0.95)

    figure(f_conn_all_mats)
    nexttile
    title(sub)
    imagesc(AEC);colorbar;
%     clim(lims)
    axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
    yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
        'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
    xticks([5, 14, 25, 37, 44, 53, 64, 76]);
    xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
        'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

    xtickangle(45)

    AEC_all_norm = cat(3,AEC_all_norm,AEC./mean(AEC(triu(ones(size(AEC)),1)==1)));
    AEC_all = cat(3,AEC_all,AEC);

    %     drawnow
end


%%
% conn_thresh = .08
% 
% % meanAEC = mean(AEC_all_norm,3);
% meanAEC = mean(AEC_all,3);
% lims = [-1,1].*max(abs(meanAEC(:)));
% % lims = [-0.0680, 0.0680];
% n_top = 150;
% n_top =sum(meanAEC(:) > conn_thresh)/2;
% perctl = 1- n_top/(length(meanAEC)*(length(meanAEC)-1)*0.5);
% figure()
% set(gcf,'Position',[326 297 915 682])
% 
% subplot(221)
% imagesc(meanAEC);colorbar;
% clim(lims)
% axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
% yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
%     'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
% xticks([5, 14, 25, 37, 44, 53, 64, 76]);
% xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
%     'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
% 
% xtickangle(45)
% sgtitle(sprintf('Mean AEC %d-%dHz (top %d connections)',hp,lp,round((1-perctl)*3003)))
% 
% subplot(222)
% go_netviewer_perctl_lim(meanAEC,perctl,lims(2))
% view([0,0])
% subplot(224)
% go_netviewer_perctl_lim(meanAEC,perctl,lims(2))
% subplot(223)
% go_netviewer_perctl_lim(meanAEC,perctl,lims(2))
% view([-90,0])
% drawnow
% 
% 
% sgtitle(sprintf('Mean AEC %d-%dHz (top %d connections)',hp,lp,round((1-perctl)*3003)))


%% kids

project_dir_kids =  '.\Children\';
% conwin = [3,3.5];
% extension_T = '_3-3.5bl';
load(['sub_info.mat'],'KidsDatasets')

extension_T = '';

extension_AEC = '';
AEC_all_kids = [];
AEC_all_norm_kids = [];
% good_subs_kids = [1:13,16,17,19,20,21:25,27];
good_subs_kids = [1:27];

f_conn_all_kids = figure;f_conn_all_kids.Color = 'w';set(f_conn_all_kids,"Units","normalized","Position",[0,0,1,1])
TL = tiledlayout('flow','TileSpacing','Compact');
f_conn_all_mats_kids = figure;f_conn_all_mats_kids.Color = 'w';set(f_conn_all_mats_kids,"Units","normalized","Position",[0,0,1,1])
TL2 = tiledlayout('flow','TileSpacing','Compact');
kids_ages = [];
for sub_i = good_subs_kids
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    AEC_results_kids.ID(height(AEC_results_kids)+1) = {sub};

    exp_type = '_task-braille';
    ses = '001';
    run = 'run-001';
    kids_ages = cat(1,kids_ages,KidsDatasets.Age(startsWith(KidsDatasets.SubjID,sub)));

    filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];

    datadir = [project_dir_kids,'Data',filesep,'BIDS',filesep];

    path_AEC = [datadir,'derivatives',filesep,'AEC',extension_AEC,filesep,'sub-',sub,filesep];
    path_Tstat = [datadir,'derivatives',filesep,'Tstats',extension_T,filesep,'sub-',sub,filesep];

    files_AEC = [filename,'_AEC'];

    load(sprintf('%s%s_%d_%d_Hz_Z.mat',path_AEC,files_AEC,hp,lp))

    figure(f_conn_all_kids)
    nexttile
    title(sub)
    go_netviewer_perctl(AEC,0.95)

    figure(f_conn_all_mats_kids)
    nexttile
    title(sub)
    imagesc(AEC);colorbar;

    AEC_all_norm_kids = cat(3,AEC_all_norm_kids,AEC./mean(AEC(triu(ones(size(AEC)),1)==1)));
    AEC_all_kids = cat(3,AEC_all_kids,AEC);

    %     drawnow
end
%% PLOTS

% conn_thresh = 0.04

zscore_ = @(x) (x - mean(x,'all','omitnan'))./std(x,[],'all','omitnan')

%%% adults
% meanAEC = mean(AEC_all_norm,3);
meanAEC = (mean(AEC_all,3));
lims = [-1,1].*max(abs(meanAEC(:)));
% conn_thresh = 0.6*lims(2)
% lims = [-0.0680, 0.0680];
n_top = 150;
% n_top =sum(meanAEC(:) > conn_thresh)/2;
perctl = 1- n_top/(length(meanAEC)*(length(meanAEC)-1)*0.5);
figure()
set(gcf,'Position',[326 297 915 682])

subplot(221)
imagesc(meanAEC);colorbar;
clim(lims)
axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

xtickangle(45)
sgtitle(sprintf('Mean AEC %d-%dHz (top %d connections)',hp,lp,round((1-perctl)*3003)))

subplot(222)
go_netviewer_perctl_lim(meanAEC,perctl,lims(2))
view([0,0])
subplot(224)
go_netviewer_perctl_lim(meanAEC,perctl,lims(2))
subplot(223)
go_netviewer_perctl_lim(meanAEC,perctl,lims(2))
view([-90,0])
drawnow


sgtitle(sprintf('Mean AEC %d-%dHz (top %d connections)',hp,lp,round((1-perctl)*3003)))
%% adult gif
fig = figure();
set(gcf,'Position',[326 297 915 482])

subplot(121)
imagesc(meanAEC);colorbar;
clim(lims)
axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

xtickangle(45)
% sgtitle(sprintf('Mean AEC %d-%dHz (top %d connections)',hp,lp,round((1-perctl)*3003)))

subplot(122)
go_netviewer_perctl_lim(meanAEC,perctl,lims(2))
nImages =360;
for fi = 1:nImages
    view([90+200.*sind(mod(1*fi,180)) 30])
    pause(0.05)
    drawnow
    frame = getframe(fig);
    im{fi} = frame2im(frame);
end

filename_gif = "connectivityAnimated.gif"; % Specify the output file name
for idx = 1:nImages
    [A,map] = rgb2ind(im{idx},256);
    if idx == 1
        imwrite(A,map,filename_gif,"gif","LoopCount",Inf,"DelayTime",0.05);
    else
        imwrite(A,map,filename_gif,"gif","WriteMode","append","DelayTime",0.05);
    end
end
%%
%%% kids
% meanAEC_kids = mean(AEC_all_norm_kids,3);
meanAEC_kids = (mean(AEC_all_kids,3));
lims_kids = [-1,1].*max(abs(meanAEC_kids(:)));
% lims_kids = lims;
% n_top =sum(meanAEC_kids(:) > conn_thresh)/2;

perctl = 1- n_top/(length(meanAEC_kids)*(length(meanAEC_kids)-1)*0.5);
figure()
set(gcf,'Position',[326 297 915 682])

subplot(221)
imagesc(meanAEC_kids);colorbar;
clim(lims_kids)
axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

xtickangle(45)
sgtitle(sprintf('Mean AEC %d-%dHz (top %d connections) | Children',hp,lp,round((1-perctl)*3003)))

subplot(222)
go_netviewer_perctl_lim(meanAEC_kids,perctl,lims_kids(2))
view([0,0])
subplot(224)
go_netviewer_perctl_lim(meanAEC_kids,perctl,lims_kids(2))
subplot(223)
go_netviewer_perctl_lim(meanAEC_kids,perctl,lims_kids(2))
view([-90,0])
drawnow


sgtitle(sprintf('Mean AEC %d-%dHz (top %d connections) | Children',hp,lp,round((1-perctl)*3003)))

%%
zscore_ = @(x) (x - mean(x,'all','omitnan'))./std(x,[],'all','omitnan')
%diff
perctl = .95
n_top = 150
figure()
set(gcf,'Position',[326 297 915 682])
% difference = zscore_(meanAEC) - zscore_(meanAEC_kids) ;
difference = (meanAEC) - (meanAEC_kids) ;
subplot(221)
imagesc(difference);colorbar;
% clim(lims_kids)
axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

xtickangle(45)
sgtitle(sprintf('Diff AEC %d-%dHz (top %d connections) | Children',hp,lp,round((1-perctl)*3003)))

subplot(222)
go_netviewer_perctl_lim(difference,perctl,[])
view([0,0])
subplot(224)
go_netviewer_perctl_lim(difference,perctl,[])
subplot(223)
go_netviewer_perctl_lim(difference,perctl,[])
view([-90,0])
drawnow

sgtitle(sprintf('Diff AEC %d-%dHz (top %d connections) | Children',hp,lp,round((1-perctl)*3003)))

%%
AEC_all_wb = AEC_all;
AEC_all_kids_wb = AEC_all_kids;
load('net_mats_mask.mat', 'net_mats_mask')
nets.lat_vis = net_mats_mask(:,:,1);
nets.central_vis = net_mats_mask(:,:,2);
nets.lat_vis2 = net_mats_mask(:,:,3);
nets.bilat_parietal = net_mats_mask(:,:,4);
nets.sens_mot = net_mats_mask(:,:,5);
nets.med_parietal = net_mats_mask(:,:,6);
nets.default_mod = net_mats_mask(:,:,7);
nets.left_frontopar = net_mats_mask(:,:,8);
nets.right_frontopar = net_mats_mask(:,:,9);
nets.frontal = net_mats_mask(:,:,10);


AEC_adults.lat_vis = AEC_all.*repmat(nets.lat_vis,[1,1,size(AEC_all,3)]);
AEC_adults.central_vis  = AEC_all.*repmat(nets.central_vis,[1,1,size(AEC_all,3)]);
AEC_adults.lat_vis2 = AEC_all.*repmat(nets.lat_vis2,[1,1,size(AEC_all,3)]);
AEC_adults.bilat_parietal = AEC_all.*repmat(nets.bilat_parietal,[1,1,size(AEC_all,3)]);
AEC_adults.sens_mot = AEC_all.*repmat(nets.sens_mot,[1,1,size(AEC_all,3)]);
AEC_adults.med_parietal = AEC_all.*repmat(nets.med_parietal,[1,1,size(AEC_all,3)]);
AEC_adults.default_mod  = AEC_all.*repmat(nets.default_mod,[1,1,size(AEC_all,3)]);
AEC_adults.left_frontopar = AEC_all.*repmat(nets.left_frontopar,[1,1,size(AEC_all,3)]);
AEC_adults.right_frontopar = AEC_all.*repmat(nets.right_frontopar,[1,1,size(AEC_all,3)]);
AEC_adults.frontal = AEC_all.*repmat(nets.frontal,[1,1,size(AEC_all,3)]);

AEC_Kids.lat_vis = AEC_all_kids.*repmat(nets.lat_vis,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.central_vis  = AEC_all_kids.*repmat(nets.central_vis,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.lat_vis2 = AEC_all_kids.*repmat(nets.lat_vis2,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.bilat_parietal = AEC_all_kids.*repmat(nets.bilat_parietal,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.sens_mot = AEC_all_kids.*repmat(nets.sens_mot,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.med_parietal = AEC_all_kids.*repmat(nets.med_parietal,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.default_mod  = AEC_all_kids.*repmat(nets.default_mod,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.left_frontopar = AEC_all_kids.*repmat(nets.left_frontopar,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.right_frontopar = AEC_all_kids.*repmat(nets.right_frontopar,[1,1,size(AEC_all_kids,3)]);
AEC_Kids.frontal = AEC_all_kids.*repmat(nets.frontal,[1,1,size(AEC_all_kids,3)]);

strength_fun = @(x) squeeze(sum(sum(x,1,'omitnan'),2,'omitnan'));
Adult_str = structfun(strength_fun,AEC_adults,'UniformOutput',false);
Kids_str = structfun(strength_fun,AEC_Kids,'UniformOutput',false);


strength_adults = squeeze(nansum(nansum(AEC_all,1),2));
strength_kids = squeeze(nansum(nansum(AEC_all_kids,1),2));
strength_kids_gt4 = squeeze(nansum(nansum(AEC_all_kids(:,:,kids_ages>4),1),2));
strength_kids_lt4 = squeeze(nansum(nansum(AEC_all_kids(:,:,kids_ages<=4),1),2));

%% fill table
AEC_results_adults.conn_strength = strength_adults;
AEC_results_kids.conn_strength = strength_kids;

AEC_results_adults.conn_mat = squeeze(mat2cell(AEC_all,size(AEC_all,1),size(AEC_all,2),ones(1,size(AEC_all,3))));
AEC_results_kids.conn_mat = squeeze(mat2cell(AEC_all_kids,size(AEC_all_kids,1),size(AEC_all_kids,2),ones(1,size(AEC_all_kids,3))));
AEC_results_adults.conn_degree = squeeze(sum(AEC_all,2,'omitnan'))';
AEC_results_kids.conn_degree = squeeze(sum(AEC_all_kids,2,'omitnan'))';
AEC_results_adults = [AEC_results_adults,struct2table(Adult_str)];
AEC_results_kids = [AEC_results_kids,struct2table(Kids_str)];
save('AEC_RESULTS_badtrl.mat','AEC_results_adults','AEC_results_kids')
%%
addpath ./Violinplot-Matlab-master/
% ylims = [-0.2,0.6];
figure('Color','w','Position',[680 470 630 508])
conn=[];
conn.Adults = strength_adults;
conn.Children_all = strength_kids;
vs = violinplot(conn);
ylabel('conn strength');
% title('D2')
box off
% ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(strength_adults,strength_kids)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))


% [h,p,ci,stats] = ttest2(strength_adults,strength_kids)
% title(sprintf("p = %1.7f (2-sided T-test)",p))


figure('Color','w','Position',[680 470 630 508])

conn.Adults = strength_adults;
conn.Children_o4 = strength_kids_gt4;
conn.Children_u4 = strength_kids_lt4;
vs = violinplot(conn);
ylabel('conn strength');
% title('D2')
box off
% ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p_adult_v_all = ranksum(strength_adults,strength_kids)
p_gt4_v_adult = ranksum(strength_adults,strength_kids_gt4)
p_lt4_v_adult = ranksum(strength_adults,strength_kids_lt4)
title(sprintf("p_{Adults v >4YO} = %1.7f\n\n " + ...
    "p_{Adults v <=4yo} = %1.7f\n(Wilcoxon RS test)",p_gt4_v_adult,p_lt4_v_adult))
xticklabels({'Adults';'Children_{all}';'Children_{(>= 4 YO)}';'Children_{(> 4 YO)}'})



%% sub net strengths
ylims = [0,0.3];
% Lateral visual 1
figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Lateral visual')
net_ = (nets.lat_vis).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.lat_vis./sum(nets.lat_vis,'all');net_strength.Children = Kids_str.lat_vis./sum(nets.lat_vis,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Lateral_visual_1','png')

% central visual
figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Central visual')
net_ = (nets.central_vis).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.central_vis./sum(nets.central_vis,'all');
net_strength.Children = Kids_str.central_vis./sum(nets.central_vis,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Central_visual','png')

% lateral visual 2
figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Lateral visual 2')
net_ = (nets.lat_vis2).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.lat_vis2./sum(nets.lat_vis2,'all');
net_strength.Children = Kids_str.lat_vis2./sum(nets.lat_vis2,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Lateral_visual_2','png')

% Bilateral Parietal

figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Bilateral Parietal')
net_ = (nets.bilat_parietal).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.bilat_parietal./sum(nets.bilat_parietal,'all');
net_strength.Children = Kids_str.bilat_parietal./sum(nets.bilat_parietal,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Bilateral_parietal','png')

% Sensorimotor
figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Sensorimotor')
net_ = (nets.sens_mot).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.sens_mot./sum(nets.sens_mot,'all');
net_strength.Children = Kids_str.sens_mot./sum(nets.sens_mot,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Sensorimotor','png')

% Medial parietal

figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Medial parietal')
net_ = (nets.med_parietal).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.med_parietal./sum(nets.med_parietal,'all');
net_strength.Children = Kids_str.med_parietal./sum(nets.med_parietal,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Medial_parietal','png')

% default mode

figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Default mode')
net_ = (nets.default_mod).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.default_mod./sum(nets.default_mod,'all');
net_strength.Children = Kids_str.default_mod./sum(nets.default_mod,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Default_mode','png')

% left frontoparietal
figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('left frontoparietal')
net_ = (nets.left_frontopar).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.left_frontopar./sum(nets.left_frontopar,'all');
net_strength.Children = Kids_str.left_frontopar./sum(nets.left_frontopar,'all');

nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/left frontoparietal','png')


% right frontoparietal
figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('right frontoparietal')
net_ = (nets.right_frontopar).*1;net_(net_==0)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.right_frontopar./sum(nets.right_frontopar,'all');
net_strength.Children = Kids_str.right_frontopar./sum(nets.right_frontopar,'all');

nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/right frontoparietal','png')


% frontal
figure('Color','w','Units','centimeters','Position',[5 5 10 20])
tiledlayout('flow','TileSpacing','Compact');
nexttile
sgtitle('Frontal')
net_ = (nets.frontal).*1;net_(net_<1)=nan;
go_netviewer_gray(net_);
net_strength=[];
net_strength.Adults = Adult_str.frontal./sum(nets.frontal,'all');
net_strength.Children = Kids_str.frontal./sum(nets.frontal,'all');
nexttile
vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("p = %1.7f (Wilkoxon Sum-rank test)",p))
saveas(gcf,'sub_network_strengths/Frontal','png')

%% violin only
%% sub net strengths
ylims = [0,0.27];
% Lateral visual 1
figure('Color','w','Units','centimeters','Position',[5 11 45.7471 10])
tiledlayout(1,10,'TileSpacing','Compact');
nexttile

net_strength=[];
net_strength.Adults = Adult_str.lat_vis./sum(nets.lat_vis,'all');
net_strength.Children = Kids_str.lat_vis./sum(nets.lat_vis,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Lateral Visual\np = %1.7f ",p))


% central visual
nexttile
net_strength=[];
net_strength.Adults = Adult_str.central_vis./sum(nets.central_vis,'all');
net_strength.Children = Kids_str.central_vis./sum(nets.central_vis,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Central visual\np = %1.7f ",p))

% lateral visual 2
nexttile
net_strength=[];
net_strength.Adults = Adult_str.lat_vis2./sum(nets.lat_vis2,'all');
net_strength.Children = Kids_str.lat_vis2./sum(nets.lat_vis2,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Lateral visual 2\np = %1.7f ",p))

% Bilateral Parietal
nexttile
net_strength=[];
net_strength.Adults = Adult_str.bilat_parietal./sum(nets.bilat_parietal,'all');
net_strength.Children = Kids_str.bilat_parietal./sum(nets.bilat_parietal,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Bilateral Parietal\np = %1.7f ",p))

% Sensorimotor
nexttile
net_strength=[];
net_strength.Adults = Adult_str.sens_mot./sum(nets.sens_mot,'all');
net_strength.Children = Kids_str.sens_mot./sum(nets.sens_mot,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Sensorimotor\np = %1.7f ",p))

% Medial parietal

nexttile
net_strength=[];
net_strength.Adults = Adult_str.med_parietal./sum(nets.med_parietal,'all');
net_strength.Children = Kids_str.med_parietal./sum(nets.med_parietal,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Medial parietal\np = %1.7f ",p))

% default mode

nexttile
net_strength=[];
net_strength.Adults = Adult_str.default_mod./sum(nets.default_mod,'all');
net_strength.Children = Kids_str.default_mod./sum(nets.default_mod,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Default mode\np = %1.7f ",p))

% left frontoparietal
nexttile
net_strength=[];
net_strength.Adults = Adult_str.left_frontopar./sum(nets.left_frontopar,'all');
net_strength.Children = Kids_str.left_frontopar./sum(nets.left_frontopar,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("left frontoparietal\np = %1.7f ",p))

% right frontoparietal
nexttile
net_strength=[];
net_strength.Adults = Adult_str.right_frontopar./sum(nets.right_frontopar,'all');
net_strength.Children = Kids_str.right_frontopar./sum(nets.right_frontopar,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("right frontoparietal\np = %1.7f ",p))

% frontal
nexttile
net_strength=[];
net_strength.Adults = Adult_str.frontal./sum(nets.frontal,'all');
net_strength.Children = Kids_str.frontal./sum(nets.frontal,'all');

vs = violinplot(net_strength);
ylabel('conn strength');
% title('D2')
box off
ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
p = ranksum(net_strength.Adults,net_strength.Children)
title(sprintf("Frontal\np = %1.7f",p))
%% maximum

max_adults = squeeze(max(max(AEC_all,[],1),[],2));
max_kids = squeeze(max(max(AEC_all_kids,[],1),[],2));

addpath ./Violinplot-Matlab-master/
% ylims = [-0.2,0.6];
figure('Color','w','Position',[680 470 630 508])

conn.Adults = max_adults;
conn.Children = max_kids;
vs = violinplot(conn);
ylabel('conn max');
% title('D2')
box off
% ylim(ylims)
xlim(xlim + [-.1,.1])
ax= gca;
% ax.FontSize=FtSz;
ranksum(max_adults,max_kids)

adult_col = [0 0.4470 0.7410];
kid_col = [0.8500 0.3250 0.0980];
figure('Color','w')
sz = 35;
scatter([kids_ages],[max_kids],sz,kid_col,'filled','DisplayName','Chilren')
hold on
scatter([adults_ages],[max_adults],sz,adult_col,'filled','DisplayName','Adults')
xlabel('Age')
ylabel('Connectivity max')
[r,p] =corr(kids_ages,max_kids)
[r,p] =corr(adults_ages,max_adults)


%% age sub groups
perctl = .95
n_top = 150
mean_kids_lt4 = (mean(AEC_all_kids(:,:,kids_ages<=4),3));
mean_kids_gt4 = (mean(AEC_all_kids(:,:,kids_ages>4),3));

figure()
set(gcf,'Position',[326 297 915 682])
subplot(221)
imagesc(mean_kids_lt4);cb = colorbar;cb.FontSize=18;
clim(lims_kids)
axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

xtickangle(45)
sgtitle(sprintf('Kids age <=4 AEC %d-%dHz (top %d connections) | Children',hp,lp,round((1-perctl)*3003)))

subplot(222)
go_netviewer_perctl_lim(mean_kids_lt4,perctl,[])
view([0,0])
subplot(224)
go_netviewer_perctl_lim(mean_kids_lt4,perctl,[])
subplot(223)
go_netviewer_perctl_lim(mean_kids_lt4,perctl,[])
view([-90,0])
drawnow

%%%%%%%%%
figure()
set(gcf,'Position',[326 297 915 682])
subplot(221)
imagesc(mean_kids_gt4);cb = colorbar;cb.FontSize=18;
clim(lims_kids)
axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

xtickangle(45)
sgtitle(sprintf('Kids age >4 AEC %d-%dHz (top %d connections) | Children',hp,lp,round((1-perctl)*3003)))

subplot(222)
go_netviewer_perctl_lim(mean_kids_gt4,perctl,[])
view([0,0])
subplot(224)
go_netviewer_perctl_lim(mean_kids_gt4,perctl,[])
subplot(223)
go_netviewer_perctl_lim(mean_kids_gt4,perctl,[])
view([-90,0])
drawnow

%% %%%%  ADULTS
meanAEC = (mean(AEC_all,3));
lims = [-1,1].*max(abs(meanAEC(:)));
figure()
set(gcf,'Position',[326 297 915 682])
subplot(221)
imagesc(meanAEC);cb = colorbar;cb.FontSize=18;
clim(lims)
axis square; yticks([5, 14, 25, 37, 44, 53, 64, 76]);
yticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});
xticks([5, 14, 25, 37, 44, 53, 64, 76]);
xticklabels({'L.M.Frontal', 'L.P.Motor', 'L.Calcarine',...
    'L.Cingulum', 'R.M.Frontal', 'R.P.Motor', 'R.Calcarine', 'R.Cingulum'});

xtickangle(45)
sgtitle(sprintf('Adults AEC %d-%dHz (top %d connections) ',hp,lp,round((1-perctl)*3003)))

subplot(222)
go_netviewer_perctl_lim(meanAEC,perctl,[])
view([0,0])
subplot(224)
go_netviewer_perctl_lim(meanAEC,perctl,[])
subplot(223)
go_netviewer_perctl_lim(meanAEC,perctl,[])
view([-90,0])
drawnow
