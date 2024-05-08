clearvars
addpath ./Violinplot-Matlab-master/

load TFS_RESULTS_BETA.mat
load AEC_RESULTS.mat
load HMM_RESULTS.mat
load EVOKED_RESULTS.mat
results_dir = '.\Figs\';
if not(all(startsWith(hmm_results_kids.ID,AEC_results_kids.ID)) & ...
all(startsWith(hmm_results_kids.ID,TFS_results_kids.ID)) & ...
all(startsWith(hmm_results_adults.ID,TFS_results_adults.ID)) & ...
all(startsWith(hmm_results_adults.ID,TFS_results_adults.ID)))
    error('Results not consistent!')
end


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

hmm_results_kids_srt = sortrows(hmm_results_kids,"age","ascend");
hmm_results_adults_srt = sortrows(hmm_results_adults,"age","ascend");
%%
figure
sz = ceil(sqrt(27));
for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    curr_TFS = TFS_results_kids.TFS_D2{startsWith(TFS_results_kids.ID,sub)};
    subplot(sz,sz,sub_i)
    plot_TFS(curr_TFS,0.3)
    title(sprintf('Sub %s: age=%1.2f',sub,hmm_results_kids.age(startsWith(hmm_results_kids.ID,sub))))
end
sgtitle('D2')

figure
sz = ceil(sqrt(27));
for sub_i = 1:27
    sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'
    curr_TFS = TFS_results_kids.TFS_D5{startsWith(TFS_results_kids.ID,sub)};
    subplot(sz,sz,sub_i)
    plot_TFS(curr_TFS,0.3)
    title(sprintf('Sub %s: age=%1.2f',sub,hmm_results_kids.age(startsWith(hmm_results_kids.ID,sub))))
end
sgtitle('D5')

%%
do_evoked = 0;
f = figure('Units','centimeters')
set(f,'Color','w')
f.Position([3,4]) = [8.9,6.935];
sub_i = 23
sub = sprintf('%3d',sub_i);sub(sub == ' ') = '0'

mean_evoked_D2 = EVOKED_results_kids.evoked_D2(startsWith(EVOKED_results_kids.ID,sub),:);
mean_evoked_D5 = EVOKED_results_kids.evoked_D5(startsWith(EVOKED_results_kids.ID,sub),:);
evoked_offset = 40;
evoked_scale = 1/50;
mean_evoked_sc_D2 = (mean_evoked_D2.*evoked_scale) + evoked_offset;
mean_evoked_sc_D5 = (mean_evoked_D5.*evoked_scale) + evoked_offset;

time = linspace(0,size(mean_evoked_D2,2)./1200,size(mean_evoked_D2,2));
fre = 3:2:50;

ax(1) = subplot(2,1,1)
curr_TFS = TFS_results_kids.TFS_D2{startsWith(TFS_results_kids.ID,sub)};
plot_TFS(curr_TFS,0.3)
cbar_lim = 0.3;
caxis([-cbar_lim cbar_lim])
axis fill
xline(ax(1),0.132,'k','LineWidth',0.5)

if do_evoked
hold on
plot(ax(1),time,mean_evoked_sc_D2,'k')
yline(ax(1),evoked_offset,'k')

yyaxis right
ax2=gca;
ax2.YColor = 'k';
ylim([(fre(1)-fre(end)+fre(end)-evoked_offset)/evoked_scale,10/evoked_scale])
yticks([-10/evoked_scale,0,10/evoked_scale])
yline(-10/evoked_scale,'--k','LineWidth',0.01)
yl=ylabel('Amp. (A.U.)');%yl.Position(2)=0;
end

ax(2) = subplot(2,1,2)
curr_TFS = TFS_results_kids.TFS_D5{startsWith(TFS_results_kids.ID,sub)};
plot_TFS(curr_TFS,0.3)
caxis([-cbar_lim cbar_lim])
cb = colorbar; 
delete(cb.Label)%.String = 'Fractional change';
axis fill
ax(1).Position([3,4]) = ax(2).Position([3,4])
cb.Position([1,2,4]) = [0.865,0.15,0.83];
ax(2).Position([3,4]) = ax(1).Position([3,4])
set([ax,cb],'FontSize',9)
ax(1).Position(2) = ax(1).Position(2) + 0.06
ax(2).Position(2) = ax(2).Position(2) + 0.04

ax(1).Position(3) = ax(1).Position(3) + 0.1
ax(2).Position(3) = ax(2).Position(3) + 0.1
hold on
xline(ax(2),0.132,'k','LineWidth',0.5)

if do_evoked
plot(ax(2),time,mean_evoked_sc_D5,'k')
yline(ax(2),evoked_offset,'k')

yyaxis right
ax2=gca;
ax2.YColor = 'k';
ylim([(fre(1)-fre(end)+fre(end)-evoked_offset)/evoked_scale,10/evoked_scale])
yticks([-10/evoked_scale,0,10/evoked_scale])
yline(-10/evoked_scale,'--k','LineWidth',0.01)
yl=ylabel('Amp. (A.U.)');%yl.Position(2)=0;
end
print_path = '.\Figs\Fig2\individual\';
print([print_path,'TFSs.png'],'-dpng','-r900')

%% % -----------------------------------------------------------------------
%%% subfunctions
%%% -----------------------------------------------------------------------
function plot_TFS(TFS,cbar_lim)
fre = 3:2:50;

time = linspace(0,size(TFS,2)./1200,size(TFS,2));
pcolor(time,fre,TFS);shading interp
xlabel('Time (s)');
ylabel('Frequency (Hz)')
% cbar_lim = 0.2;
caxis([-cbar_lim cbar_lim])
% cb = colorbar('Location','southoutside'); 
% cb.Label.String = 'Fractional change';
axis fill
ylim([3,50])

end