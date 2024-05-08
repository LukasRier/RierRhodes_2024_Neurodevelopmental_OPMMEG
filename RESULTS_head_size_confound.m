clearvars
close all
clc

fieldtrip_path = [project_dir,filesep,'fieldtrip-20220906'];
if ~exist(fieldtrip_path,"dir")
    error("change 'fieldtrip_path' to valid path")
else
    addpath(fieldtrip_path)
end
addpath ./Violinplot-Matlab-master/
results_dir = '.\Figs'

script_dir = mfilename('fullpath');fname = mfilename;script_dir = script_dir(1:end-length(fname));
addpath(script_dir)
addpath([script_dir,'Beamformer',filesep,''])

ft_defaults;
 do_plot = 0;

if true
    project_dir =  '.\Adults\';

    load(['sub_info_adults.mat'],'AdultDatasets')
    AdultDatasets.Age = years(AdultDatasets.Date - AdultDatasets.DOB);
    adults_ages = [];
    adult_scalp_distances = [];
    adult_head_circs = [];
    adult_ids = [];
    for sub_i = [1:19,21:22,24:26]%1:26
        sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0';
        ses_i = 1;
        ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0';
        adults_ages = cat(1,adults_ages,AdultDatasets.Age(startsWith(AdultDatasets.SubjID,sub)));
        [median_sens_scalp_distance,max_head_circ] = sensor_distance_func(sub,ses,project_dir);

        adult_scalp_distances= cat(1,adult_scalp_distances,median_sens_scalp_distance);

        adult_head_circs= cat(1,adult_head_circs,max_head_circ);
        adult_ids= cat(1,adult_ids,sub_i);
%              input('next?')
        drawnow
    end


    %
    project_dir =  '.\Children\';
    load(['sub_info.mat'],'KidsDatasets')
    KidsDatasets.Age_yrs = years(KidsDatasets.Date - KidsDatasets.DOB);
    kids_ages = [];
    kids_scalp_distances = [];
    kids_head_circs = [];
    kids_ids = [];

    for sub_i = 1:27
        sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0';
        ses_i = 1;
        ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0';
        [median_sens_scalp_distance,max_head_circ] = sensor_distance_func(sub,ses,project_dir);
        kids_ages = cat(1,kids_ages,KidsDatasets.Age(startsWith(KidsDatasets.SubjID,sub)));

        kids_scalp_distances= cat(1,kids_scalp_distances,median_sens_scalp_distance);
        kids_head_circs= cat(1,kids_head_circs,max_head_circ);
        kids_ids= cat(1,kids_ids,sub_i);
        drawnow

    end

    if max(adult_head_circs) > max(kids_head_circs)
        [max_circ,max_i] = max(adult_head_circs);
        refsub_i = adult_ids(max_i);
        refsub = sprintf('1%2d',refsub_i);refsub(refsub == ' ') = '0';
    end

    %%
    mean_sens_dist.aChildren = kids_scalp_distances.*1e3;
    mean_sens_dist.bAdults = adult_scalp_distances.*1e3;
    mean_sens_dist.cAll = [kids_scalp_distances;adult_scalp_distances].*1e3;
    figure
    vs = violinplot(mean_sens_dist);
    [vs.ShowMean] = deal(1);
    [vs.ShowBox] = deal(0);
    xticklabels({'Children','Adults','All'})
    set(gcf,'Color','w')
    ylabel("Mean sensor-to-scalp distance (mm)")

    fig = figure;
    scatter(kids_ages,kids_scalp_distances.*1e3,'k^')
    hold on
    scatter(adults_ages,adult_scalp_distances.*1e3,'ko')

    all_ages  = [kids_ages;adults_ages];
    all_dists = [kids_scalp_distances;adult_scalp_distances].*1e3;

    X = [ones(length(all_ages),1) all_ages];
    b = X\all_dists;
    all_dists_model = X*b;  
    plot(all_ages,all_dists_model,'r-')
    [rho,pval] = corr(all_ages,all_dists);

    legend({'Children', 'Adults'},'Location','southeast')

    set(gcf,'Color','w')
    ylabel("Sensor Proximity (mm)")
    xlabel("Age (years)")
    title(sprintf("\\rho = %1.3f, p = %1.3f\n" + ...
        "distance = %1.4fmm + (%1.4f mm/yr)*age",rho,pval,b(1),b(2)))

fig.Units = 'centimeters';
fwidth = 8;
fheight = 6;
ylim([0,30])
fig.Position([3,4]) = [fwidth,fheight];
fig.Position([1,2]) = [10 10];
xlim([0,1+max(all_ages)])

set(fig.Children,'FontName','Arial','FontSize',9)
print(gcf,sprintf('%s/HeadszConfound/sens_to_scalp_v_age.png',results_dir),'-dpng','-r900');
    %%

    project_dir =  '.\Adults\';

    adult_moved_scalp_distances = [];
    adult_gif_filename = ".\adult_moved.gif";
    for sub_i = [1:19,21:22,24:26]%1:26
        sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0';
        ses_i = 1;
        ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0';
        [median_sens_scalp_distance] = distance_single_helmet(sub,ses,project_dir,refsub);

        adult_moved_scalp_distances= cat(1,adult_moved_scalp_distances,median_sens_scalp_distance);
        %     input('next?')
        drawnow
        if do_plot
            frame = getframe(gcf);
            im = frame2im(frame);
            [A,map] = rgb2ind(im,256);
            if sub_i == 1
                imwrite(A,map,adult_gif_filename,"gif","LoopCount",Inf,"DelayTime",1);
            else
                imwrite(A,map,adult_gif_filename,"gif","WriteMode","append","DelayTime",1);
            end
        end
    end

    %
    project_dir =  '.\Children\';

    kids_moved_scalp_distances = [];
    kids_gif_filename = ".\kids_moved.gif";

    for sub_i = 1:27
        sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0';
        ses_i = 1;
        ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0';
        [median_sens_scalp_distance] = distance_single_helmet(sub,ses,project_dir,refsub);

        kids_moved_scalp_distances= cat(1,kids_moved_scalp_distances,median_sens_scalp_distance);

        drawnow
        if do_plot
            frame = getframe(gcf);
            im = frame2im(frame);
            [A,map] = rgb2ind(im,256);
            if sub_i == 1
                imwrite(A,map,kids_gif_filename,"gif","LoopCount",Inf,"DelayTime",1);
            else
                imwrite(A,map,kids_gif_filename,"gif","WriteMode","append","DelayTime",1);
            end
        end
    end


    %%
    mean_sens_dist.dChildrenMoved = kids_moved_scalp_distances.*1e3;
    mean_sens_dist.eAdultsMoved = adult_moved_scalp_distances.*1e3;
    mean_sens_dist.fAllMoved = [kids_moved_scalp_distances;adult_moved_scalp_distances].*1e3;
    figure
    vs = violinplot(mean_sens_dist);
    [vs.ShowMean] = deal(1);
    [vs.ShowBox] = deal(0);
    xticklabels({'Children','Adults','All','Children-moved','Adults-moved','All-moved'})
    set(gcf,'Color','w')
    ylabel("Mean sensor-to-scalp distance (mm)")

    fig = figure;
    scatter(kids_ages,kids_moved_scalp_distances.*1e3,"k^");hold on
    scatter(adults_ages,adult_moved_scalp_distances.*1e3,"ko")

    all_ages  = [kids_ages;adults_ages];
    all_dists_moved = [kids_moved_scalp_distances;adult_moved_scalp_distances].*1e3;

    X = [ones(length(all_ages),1) all_ages];
    b_moved = X\all_dists_moved;
    all_dists_moved_model = X*b_moved;
    plot(all_ages,all_dists_moved_model,'c-')

    [rho_moved,pval_moved] = corr(all_ages,all_dists_moved);

    legend({'Children', 'Adults','fit','Children_{moved}', 'Adults_{moved}','fit_{moved}'})
    set(gcf,'Color','w')
    ylabel("Mean sensor-to-scalp distance (mm)")
    xlabel("Age (years)")
    title(sprintf("\\rho = %1.3f, p = %1.3f\n" + ...
        "distance = %1.4fmm + (%1.4f mm/yr)*age\n\n" + ...
        "Moved: \\rho = %1.3f, p = %1.3f\n" + ...
        "distance = %1.4fmm + (%1.4f mm/yr)*age\n", ...
        rho,pval,b(1),b(2),rho_moved,pval_moved,b_moved(1),b_moved(2)))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%% CTF test
project_dir =  '.\Adults\';
refsub = '109';
refmrifile = ['.\Adults\sub-',refsub,'\ses-001\anat\',...
    'sub-',refsub,'_anat.nii'];
if ~exist("refmri_vox2ctf_transform.mat")
    refmri = ft_read_mri(refmrifile);

    refmri = ft_determine_coordsys(refmri, 'interactive', 'yes');
    cfg          = [];
    cfg.method   = 'interactive';
    cfg.coordsys = 'ctf';
    refmri          = ft_volumerealign(cfg, refmri);

    cfg            = [];
    cfg.resolution = 1;
    cfg.dim        = [256 256 256];
    mri256            = ft_volumereslice(cfg, refmri);

    vox2ctf = mri256.transform;
    fiducials = refmri.cfg.fiducial;
    save("refmri_vox2ctf_transform.mat",'mri256','vox2ctf','fiducials')
else
    load("refmri_vox2ctf_transform.mat",'mri256','vox2ctf','fiducials')
    refmri =true;
end

cfg = [];
cfg.output    = {'brain' 'scalp' 'skull'};
%cfg.scalpthreshold = 0.01; % 0.1 by default
segmented_ref_mri = ['refmri_segmented.mat'];
if ~exist(segmented_ref_mri)
    segmentedmri  = ft_volumesegment(cfg, mri256);
    save(segmented_ref_mri,"segmentedmri")
else
    load(segmented_ref_mri,'segmentedmri')
end
segmented_ref_mri_mesh = ['refmri_segmented_meshes.mat'];
if ~exist(segmented_ref_mri_mesh)
    cfg = [];
    cfg.tissue = {'brain' 'scalp' 'skull'};
    cfg.numvertices = [5000 5000 5000];

    mesh2 = ft_prepare_mesh(cfg,segmentedmri);
    mesh1 = ft_convert_units(mesh2,'m');
    for n = 1:size(mesh1,2)
        meshes_ref(n).pnt = mesh1(n).pos;
        meshes_ref(n).tri = mesh1(n).tri;
        meshes_ref(n).unit = mesh1(n).unit;
        meshes_ref(n).name = cfg.tissue{n};
    end
    save(segmented_ref_mri_mesh,"meshes_ref")
else
    load(segmented_ref_mri_mesh,"meshes_ref")
end
meshes_ref = ft_convert_units(meshes_ref,'m');

hdr = ft_read_header('.\exampleData.ds'); %any ctf dataset with fiducial coils
channel_pos = hdr.grad.chanpos(startsWith(hdr.grad.chantype,'meggrad'),:)./100;

figure
ft_plot_mesh(meshes_ref,'facecolor',[0 0 .5],'facealpha',.1,'edgecolor','none')
view([130,30])
fig = gcf;
fig.Color = [1,1,1];
hold on
plot3(channel_pos(:,1),channel_pos(:,2),channel_pos(:,3),'ko')

%%
adult_gif_filename = ".\kids_moved_ctf.gif";
adult_moved_scalp_distances_ctf = [];
for sub_i = [1:19,21:22,24:26]%1:26
    sub = sprintf('1%2d',sub_i);sub(sub == ' ') = '0';
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0';
    [median_sens_scalp_distance] = distance_ctf(sub,ses,project_dir,refsub,meshes_ref,refmri,channel_pos);

    adult_moved_scalp_distances_ctf= cat(1,adult_moved_scalp_distances_ctf,median_sens_scalp_distance);

    drawnow
    if do_plot
        frame = getframe(gcf);
        im = frame2im(frame);
        [A,map] = rgb2ind(im,256);
        if sub_i == 1
            imwrite(A,map,adult_gif_filename,"gif","LoopCount",Inf,"DelayTime",1);
        else
            imwrite(A,map,adult_gif_filename,"gif","WriteMode","append","DelayTime",1);
        end
    end
end

%
project_dir =  '.\Children\';

kids_moved_scalp_distances_ctf = [];
kids_gif_filename = ".\kids_moved_ctf.gif";

for sub_i = 1:27
    sub = sprintf('0%2d',sub_i);sub(sub == ' ') = '0';
    ses_i = 1;
    ses = sprintf('%3d',ses_i);ses(ses == ' ') = '0';
    [median_sens_scalp_distance] = distance_ctf(sub,ses,project_dir,refsub,meshes_ref,refmri,channel_pos);

    kids_moved_scalp_distances_ctf= cat(1,kids_moved_scalp_distances_ctf,median_sens_scalp_distance);

    drawnow
    if do_plot
        frame = getframe(gcf);
        im = frame2im(frame);
        [A,map] = rgb2ind(im,256);
        if sub_i == 1
            imwrite(A,map,kids_gif_filename,"gif","LoopCount",Inf,"DelayTime",1);
        else
            imwrite(A,map,kids_gif_filename,"gif","WriteMode","append","DelayTime",1);
        end
    end
end

mean_sens_dist.gChildren_ctf = kids_moved_scalp_distances_ctf.*1e3;
mean_sens_dist.hAdults_ctf = adult_moved_scalp_distances_ctf.*1e3;
mean_sens_dist.iAll_ctf = [kids_moved_scalp_distances_ctf;adult_moved_scalp_distances_ctf].*1e3;
figure()
vs = violinplot(mean_sens_dist);
[vs.ShowMean] = deal(1);
[vs.ShowBox] = deal(0);
xticklabels({'Children','Adults','All',...
    'Children-moved','Adults-moved','All-moved',...
    'Children_{CTF}','Adults_{CTF}','All_{CTF}'})
set(gcf,'Color','w')
ylabel("Mean sensor-to-scalp distance (mm)")

figure()
scatter(kids_ages,kids_moved_scalp_distances_ctf.*1e3,'k^')
hold on
scatter(adults_ages,adult_moved_scalp_distances_ctf.*1e3,'ko')

all_ages  = [kids_ages;adults_ages];
all_dists = [kids_moved_scalp_distances_ctf;adult_moved_scalp_distances_ctf].*1e3;

X = [ones(length(all_ages),1) all_ages];
b = X\all_dists;
all_dists_model = X*b;
plot(all_ages,all_dists_model,'r-')
[rho,pval] = corr(all_ages,all_dists);

legend({'Children', 'Adults'})

set(gcf,'Color','w')
fig.Units = 'centimeters';
fwidth = 8;
fheight = 6;

fig.Position([3,4]) = [fwidth,fheight];
fig.Position([1,2]) = [10 10];
xlim([0,1+max(adults_ages)])

ylabel("CTF | Mean sensor-to-scalp distance (mm)")
xlabel("Age (years)")
title(sprintf("\\rho = %1.3f, p = %1.3f\n" + ...
    "distance = %1.4fmm + (%1.4f mm/yr)*age",rho,pval,b(1),b(2)))
%%
figure()
scatter(kids_head_circs,kids_moved_scalp_distances_ctf.*1e3)
hold on
scatter(adult_head_circs,adult_moved_scalp_distances_ctf.*1e3)

all_head_circs  = [kids_head_circs;adult_head_circs];
all_dists = [kids_moved_scalp_distances_ctf;adult_moved_scalp_distances_ctf].*1e3;

X = [ones(length(all_head_circs),1) all_head_circs];
b = X\all_dists;
all_dists_model = X*b;
plot(all_head_circs,all_dists_model,'r-')
[rho,pval] = corr(all_head_circs,all_dists);

legend({'Children', 'Adults'})

set(gcf,'Color','w')

fig.Units = 'centimeters';
fwidth = 8;
fheight = 6;

fig.Position([3,4]) = [fwidth,fheight];
fig.Position([1,2]) = [10 10];
xlim([0,1+max(adult_ages)])
ylabel("CTF | Mean sensor-to-scalp distance (mm)")
xlabel("Head circumference (mm)")
title(sprintf("\\rho = %1.3f, p = %1.5f\n" + ...
    "distance = %1.4fmm + (%1.4f mm/yr)*age",rho,pval,b(1),b(2)))
%%

%%

fig=figure();
set(fig,'Color','w')
fig.Units = 'centimeters';
fwidth = 16;
fheight = 12;
fig.Position([3,4]) = [fwidth,fheight];
subplot(2,2,1)

scatter(kids_ages,kids_scalp_distances.*1e3,'k^')
hold on
scatter(adults_ages,adult_scalp_distances.*1e3,'ko')

all_ages  = [kids_ages;adults_ages];
all_dists = [kids_scalp_distances;adult_scalp_distances].*1e3;

X = [ones(length(all_ages),1) all_ages];
b = X\all_dists;
all_dists_model = X*b;
plot(all_ages,all_dists_model,'r-')
[rho,pval] = corr(all_ages,all_dists);

legend({'Children', 'Adults'},'box','on','Location','northeast')

set(gcf,'Color','w')
ylabel("Sensor Proximity (mm)")
xlabel("Age (years)")
ylim([0,30])
text(1,4,sprintf("\\rho = %1.2f, p = %1.1e\n" + ...
    "y = %1.1fmm + (%1.2f)*x",rho,pval,b(1),b(2)))

subplot(2,2,2)
scatter(kids_ages,kids_moved_scalp_distances.*1e3,'k^')
hold on
scatter(adults_ages,adult_moved_scalp_distances.*1e3,'ko')

all_ages  = [kids_ages;adults_ages];
all_dists_moved = [kids_moved_scalp_distances;adult_moved_scalp_distances].*1e3;

X = [ones(length(all_ages),1) all_ages];
b_moved = X\all_dists_moved;
all_dists_moved_model = X*b_moved;
plot(all_ages,all_dists_moved_model,'r')

[rho_moved,pval_moved] = corr(all_ages,all_dists_moved);

% legend({'Children', 'Adults'})
set(gcf,'Color','w')
ylabel("Sensor Proximity (mm)")
xlabel("Age (years)")

ylim([0,30])
text(1,4,sprintf("\\rho = %1.2f, p = %1.1e\n" + ...
    "y = %1.1fmm + (%1.2f)*x",rho_moved,pval_moved,b_moved(1),b_moved(2)))
% ylim([0,30])

subplot(2,2,3)
scatter(kids_head_circs,kids_scalp_distances.*1e3,'k^')
hold on
scatter(adult_head_circs,adult_scalp_distances.*1e3,'ko')

all_head_circs  = [kids_head_circs;adult_head_circs];
all_dists = [kids_scalp_distances;adult_scalp_distances].*1e3;

X = [ones(length(all_head_circs),1) all_head_circs];
b = X\all_dists;
all_dists_model = X*b;
plot(all_head_circs,all_dists_model,'r-')
[rho,pval] = corr(all_head_circs,all_dists);
% ylim([0,30])
% legend({'Children', 'Adults'})

set(gcf,'Color','w')
ylabel("Sensor Proximity (mm)")
xlabel("Head circumference (mm)")
ylim([0,30])
text(490,5,sprintf("\\rho = %1.2f, p = %1.1e\n" + ...
    "y = %1.1fmm + (%1.2f)*x",rho,pval,b(1),b(2)))

subplot(2,2,4)

scatter(kids_head_circs,kids_moved_scalp_distances.*1e3,'k^')
hold on
scatter(adult_head_circs,adult_moved_scalp_distances.*1e3,'ko')

all_head_circs  = [kids_head_circs;adult_head_circs];
all_dists = [kids_moved_scalp_distances;adult_moved_scalp_distances].*1e3;

X = [ones(length(all_head_circs),1) all_head_circs];
b = X\all_dists;
all_dists_model = X*b;
plot(all_head_circs,all_dists_model,'r-')
[rho,pval] = corr(all_head_circs,all_dists);

% legend({'Children', 'Adults'})

set(gcf,'Color','w')
ylabel("Sensor Proximity (mm)")
xlabel("Head circumference (mm)")
ylim([0,30])
text(490,5,sprintf("\\rho = %1.2f, p = %1.1e\n" + ...
    "y = %1.1fmm + (%1.2f)*x",rho,pval,b(1),b(2)))

print(gcf,sprintf('%s/HeadszConfound/All.png',results_dir),'-dpng','-r900');

%%
% Significant slope test
all_head_circs  = [kids_head_circs;adult_head_circs];
all_dists = [kids_scalp_distances;adult_scalp_distances].*1e3;
all_dists_moved = [kids_moved_scalp_distances;adult_moved_scalp_distances].*1e3;

X = [ones(length(all_head_circs),1) all_head_circs];

b = X\all_dists;
b_moved = X\all_dists_moved;

slope_diff = b(2) - b_moved(2);
Nperm = 10000;
slope_diff_perm = zeros(1,Nperm);
for perm_i = 1:Nperm
    grp = randi(2,size(all_dists))==1;
    all_dists_perm = all_dists;
    all_dists_moved_perm = all_dists_moved;

    all_dists_perm(grp) = all_dists_moved(grp);
    all_dists_moved_perm(grp) = all_dists(grp);
    b = X\all_dists_perm;
    b_moved = X\all_dists_moved_perm;
    slope_diff_perm(perm_i) = b(2) - b_moved(2);
end
sig_cutoff = prctile(slope_diff_perm,97.5)

figure('Color','w')
histogram(slope_diff_perm,100)
hold on
tru_l=xline(slope_diff,'r','LineWidth',2);
xlabel("Slope difference")

patch([sig_cutoff, sig_cutoff, max(xlim), max(xlim)],...
    [min(ylim), max(ylim), max(ylim), min(ylim)], [1 0.0 0],...
    "FaceAlpha",0.1,"EdgeColor","None")
legend({"Permutations","Actual value","p<0.025"},'Location','Northwest')
p_perm = sum(slope_diff_perm > slope_diff)/Nperm

% ANCOVA
moved = [repmat("OPM",length(all_dists),1);repmat("OPM_{moved}",length(all_dists_moved),1)];
head_circumference = [all_head_circs;all_head_circs];
sens_scalp_dists = [all_dists;all_dists_moved];

aoctool(head_circumference,sens_scalp_dists,moved)


figure
scatter(all_ages,all_head_circs)
hold on
X = [ones(length(all_ages),1) all_ages];

b = X\all_head_circs;
all_circs_model = X*b;
plot(all_ages,all_circs_model,'r-')
[rho,pval] = corr(all_ages,all_head_circs);
% ylim([0,30])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [average_sens_scalp_distance,max_head_circ] = sensor_distance_func(sub,ses,project_dir)
 do_plot = 0;
run = 'run-001';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datadir = [project_dir,'Data',filesep,'BIDS',filesep];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_type = '_task-braille';
filename = ['sub-',sub,'_ses-',ses,exp_type,'_',run];
path_main = [datadir,'sub-',sub,filesep,'ses-',ses,filesep];


path_meg_data = [path_main,'meg',filesep];

path_meshes = [datadir,'derivatives',filesep,'sourcespace',filesep,'sub-',sub,filesep];
files_meshes = ['sub-',sub,'_meshes.mat'];

path_mri = [path_main,'anat',filesep];
files_mri = ['sub-',sub,'_anat.nii'];
S.mri_file = [path_mri,files_mri];
files_outskin_mask = ['sub-',sub,'_anat_brain_outskin_mask.nii.gz'];
path_helmet = [datadir,'derivatives',filesep,'helmet',filesep,'sub-',sub,filesep];
files_helmet_info = dir([path_helmet,'*.mat']);files_helmet_info=files_helmet_info.name;

files_channels = [filename,'_channels.tsv'];

cd(path_meg_data)

%helmet info
load([path_helmet,files_helmet_info]);


ch_table = readtable([path_meg_data,files_channels],'FileType','text','Delimiter','tab');
ch_table.isx = endsWith(ch_table.name,'X');
ch_table.isy = endsWith(ch_table.name,'Y');
ch_table.isz = endsWith(ch_table.name,'Z');
ch_table.slot_no = zeros(height(ch_table),1);


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


% Load meshes

load([path_meshes,files_meshes],'meshes');

meshes = ft_convert_units(meshes,'m');

scalp_mesh = meshes(startsWith({meshes.name},'scalp'));
% scalp_mesh = meshes(startsWith({meshes.name},'brain'));
sens_locs = S.sensor_info.pos(ch_table.isx,:);

Mdl = KDTreeSearcher(scalp_mesh.pnt);

[idx, sens_scalp_distances] = knnsearch(Mdl,sens_locs);

average_sens_scalp_distance = mean(sens_scalp_distances);
% average_sens_scalp_distance = median(sens_scalp_distances);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% make a plot of the geometry...


outskin_mask = ft_read_mri([path_mri,files_outskin_mask]);
outskin_mask = ft_convert_units(outskin_mask,'m');

circumfs = zeros(1,outskin_mask.dim(3));
for sli = 1:outskin_mask.dim(3)
    perim = regionprops(outskin_mask.anatomy(:,:,sli),'perimeter');
    if ~isempty(perim)
        circumfs(sli) = perim.Perimeter;
    end
end

[max_head_circ ,max_sl] = max(circumfs);
if do_plot
    figure
    ft_plot_mesh(meshes,'facecolor',[.5 .5 .5],'facealpha',.3,'edgecolor','none')
    hold on
    view([130,30])
    fig = gcf;
    fig.Color = [1,1,1];
    plot3(S.sensor_info.pos(:,1),S.sensor_info.pos(:,2),S.sensor_info.pos(:,3),'o')
    sens2scalp_vec = scalp_mesh.pnt(idx,:) - sens_locs;
    quiver3(sens_locs(:,1),sens_locs(:,2),sens_locs(:,3),...
        sens2scalp_vec(:,1),sens2scalp_vec(:,2),sens2scalp_vec(:,3),'off')
    figure
    imagesc(outskin_mask.anatomy(:,:,max_sl))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


function [median_sens_scalp_distance] = distance_single_helmet(sub,ses,project_dir,refsub)
 do_plot = 0;
run = 'run-001';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datadir = [project_dir,'Data',filesep,'BIDS',filesep];
refdatadir = ['R:\DRS-KidsOPM\Paediatric_OPM_Notts_AdultData\','Data',filesep,'BIDS',filesep];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_type = '_task-braille';
filename = ['sub-',refsub,'_ses-',ses,exp_type,'_',run];
path_main = [refdatadir,'sub-',refsub,filesep,'ses-',ses,filesep];


path_meg_data = [path_main,'meg',filesep];

path_meshes = [refdatadir,'derivatives',filesep,'sourcespace',filesep,'sub-',refsub,filesep];
files_meshes = ['sub-',refsub,'_meshes.mat'];


path_moved_mri = [datadir,'derivatives',filesep,'headsize_confound',filesep,'sub-',sub,filesep];
files_moved_mri = ['sub-',sub,'_to_sub-',refsub,'_rigid.nii.gz'];

path_helmet = [refdatadir,'derivatives',filesep,'helmet',filesep,'sub-',refsub,filesep];
files_helmet_info = dir([path_helmet,'*.mat']);files_helmet_info=files_helmet_info.name;

files_channels = [filename,'_channels.tsv'];

cd(path_meg_data)

%helmet info
load([path_helmet,files_helmet_info])


ch_table = readtable([path_meg_data,files_channels],'FileType','text','Delimiter','tab');
ch_table.isx = endsWith(ch_table.name,'X');
ch_table.isy = endsWith(ch_table.name,'Y');
ch_table.isz = endsWith(ch_table.name,'Z');
ch_table.slot_no = zeros(height(ch_table),1);


%% sensor info
S.sensor_info.pos = [ch_table.Px,ch_table.Py,ch_table.Pz]./100;
S.sensor_info.ors = [ch_table.Ox,ch_table.Oy,ch_table.Oz];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

xfm = load(sprintf('sub-%s_ses-%s%s_%s_sens2template_transform.txt',refsub,ses,exp_type,run));
xfm = reshape(xfm,4,4);
xfm_rot = xfm(1:3,1:3);
xfm_translation = xfm(1:3,4)';
S.sensor_info.pos = (xfm_rot*S.sensor_info.pos' + xfm_translation')';
S.sensor_info.ors = (xfm_rot*S.sensor_info.ors')';


% Load meshes

load([path_meshes,files_meshes],'meshes');

meshes = ft_convert_units(meshes,'m');

% scalp_mesh = meshes(startsWith({meshes.name},'scalp'));
scalp_mesh = meshes(startsWith({meshes.name},'brain'));
sens_locs = S.sensor_info.pos(ch_table.isx,:);



mri = ft_read_mri([path_moved_mri,files_moved_mri]);

cfg = [];
cfg.output    = {'brain' 'scalp' 'skull'};
%cfg.scalpthreshold = 0.01; % 0.1 by default
segmented_moved_mri = [path_moved_mri,files_moved_mri(1:end-7),'_segmented.mat'];
if ~exist(segmented_moved_mri)
    segmentedmri  = ft_volumesegment(cfg, mri);
    save(segmented_moved_mri,"segmentedmri")
else
    load(segmented_moved_mri)
end
segmented_moved_mri_mesh = [path_moved_mri,files_moved_mri(1:end-7),'_segmented_meshes.mat'];
if ~exist(segmented_moved_mri_mesh)
    cfg = [];
    cfg.tissue = {'brain' 'scalp' 'skull'};
    cfg.numvertices = [5000 5000 5000];

    mesh2 = ft_prepare_mesh(cfg,segmentedmri);
    mesh1 = ft_convert_units(mesh2,'m');
    for n = 1:size(mesh1,2)
        meshes_moved(n).pnt = mesh1(n).pos;
        meshes_moved(n).tri = mesh1(n).tri;
        meshes_moved(n).unit = mesh1(n).unit;
        meshes_moved(n).name = cfg.tissue{n};
    end
    save(segmented_moved_mri_mesh,"meshes_moved")
else
    load(segmented_moved_mri_mesh,"meshes_moved")
end
meshes_moved = ft_convert_units(meshes_moved,'m');

scalp_mesh_moved = meshes_moved(startsWith({meshes_moved.name},'scalp'));
% scalp_mesh_moved = meshes_moved(startsWith({meshes_moved.name},'brain'));


Mdl = KDTreeSearcher(scalp_mesh_moved.pnt);

[idx, sens_scalp_distances] = knnsearch(Mdl,sens_locs);

% median_sens_scalp_distance = median(sens_scalp_distances);
median_sens_scalp_distance = mean(sens_scalp_distances);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% make a plot of the geometry...
if do_plot
    figure
    ft_plot_mesh(meshes_moved,'facecolor',[0.5 0.5 .5],'facealpha',.3,'edgecolor','none')
    hold on
    ft_plot_mesh(scalp_mesh,'facecolor',[0 0 .5],'facealpha',.1,'edgecolor','none')

    view([130,30])
    fig = gcf;
    fig.Color = [1,1,1];
    plot3(S.sensor_info.pos(:,1),S.sensor_info.pos(:,2),S.sensor_info.pos(:,3),'o')
    sens2scalp_vec = scalp_mesh_moved.pnt(idx,:) - sens_locs;
    quiver3(sens_locs(:,1),sens_locs(:,2),sens_locs(:,3),...
        sens2scalp_vec(:,1),sens2scalp_vec(:,2),sens2scalp_vec(:,3),'off')
    set(gcf,'Name',['sub ',sub])
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


function [median_sens_scalp_distance] = distance_ctf(sub,ses,project_dir,refsub,meshes_ref,refmri,sens_locs)
 do_plot = 0;
run = 'run-001';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datadir = [project_dir,'Data',filesep,'BIDS',filesep];
refdatadir = ['.\Adults\Data',filesep,'BIDS',filesep];

% scalp_mesh_ref = meshes_ref(startsWith({meshes_ref.name},'scalp'));
scalp_mesh_ref = meshes_ref(startsWith({meshes_ref.name},'brain'));

path_moved_mri = [datadir,'derivatives',filesep,'headsize_confound',filesep,'sub-',sub,filesep];
files_moved_mri = ['sub-',sub,'_to_sub-',refsub,'_rigid.nii.gz'];

%cfg.scalpthreshold = 0.01; % 0.1 by default
segmented_moved_mri256 = [path_moved_mri,files_moved_mri(1:end-7),'_segmented_256.mat'];
if ~exist(segmented_moved_mri256)
    mri = ft_read_mri([path_moved_mri,files_moved_mri]);
    refmri.anatomy = mri.anatomy;


    cfg            = [];
    cfg.resolution = 1;
    cfg.dim        = [256 256 256];
    indiv_mri256            = ft_volumereslice(cfg, refmri);


    cfg = [];
    cfg.output    = {'brain' 'scalp' 'skull'};

    segmentedmri  = ft_volumesegment(cfg, indiv_mri256);
    save(segmented_moved_mri256,"segmentedmri")
else
    load(segmented_moved_mri256)
end
segmented_moved_mri_mesh256 = [path_moved_mri,files_moved_mri(1:end-7),'_segmented_meshes_256.mat'];
if ~exist(segmented_moved_mri_mesh256)
    cfg = [];
    cfg.tissue = {'brain' 'scalp' 'skull'};
    cfg.numvertices = [5000 5000 5000];

    mesh2 = ft_prepare_mesh(cfg,segmentedmri);
    mesh1 = ft_convert_units(mesh2,'m');
    for n = 1:size(mesh1,2)
        meshes_moved256(n).pnt = mesh1(n).pos;
        meshes_moved256(n).tri = mesh1(n).tri;
        meshes_moved256(n).unit = mesh1(n).unit;
        meshes_moved256(n).name = cfg.tissue{n};
    end
    save(segmented_moved_mri_mesh256,"meshes_moved256")
else
    load(segmented_moved_mri_mesh256,"meshes_moved256")
end
meshes_moved256 = ft_convert_units(meshes_moved256,'m');

% scalp_mesh_moved256 = meshes_moved256(startsWith({meshes_moved256.name},'scalp'));
scalp_mesh_moved256 = meshes_moved256(startsWith({meshes_moved256.name},'brain'));


Mdl = KDTreeSearcher(scalp_mesh_moved256.pnt);

[idx, sens_scalp_distances] = knnsearch(Mdl,sens_locs);

% median_sens_scalp_distance = median(sens_scalp_distances);
median_sens_scalp_distance = mean(sens_scalp_distances);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% make a plot of the geometry...
if do_plot
    figure
    ft_plot_mesh(meshes_moved256,'facecolor',[0.5 0.5 .5],'facealpha',.3,'edgecolor','none')
    hold on
    ft_plot_mesh(scalp_mesh_ref,'facecolor',[0 0 .5],'facealpha',.1,'edgecolor','none')

    view([130,30])
    fig = gcf;
    fig.Color = [1,1,1];
    plot3(sens_locs(:,1),sens_locs(:,2),sens_locs(:,3),'o')
    sens2scalp_vec = scalp_mesh_moved256.pnt(idx,:) - sens_locs;
    quiver3(sens_locs(:,1),sens_locs(:,2),sens_locs(:,3),...
        sens2scalp_vec(:,1),sens2scalp_vec(:,2),sens2scalp_vec(:,3),'off')
    set(gcf,'Name',['sub ',sub])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end