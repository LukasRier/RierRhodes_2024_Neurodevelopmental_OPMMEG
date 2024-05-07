function [Lead_fields, L_reshaped] = Forward_OPM_shell_FT(mri_file,sensor_info,sourcepos)
% Use FieldTrip to create a single shell lead field model
% mri_file = .mri file
% sensor_info = sensor positions (Nx3), orientations(Nx3), and labels (Nx1)
% skanect_file = where to save/load things to/from

%% Load OPM sensor positions
grad.coilpos = sensor_info.pos.*1;
grad.coilori = sensor_info.ors;
grad.label = sensor_info.label;
grad.chanpos = grad.coilpos;
grad.units = 'm';

%% Create head model
[path.mri, file_name.mri] = fileparts(mri_file)
sub = char(regexp(file_name.mri,'sub-\d\d\d','match')); %find sub-xyz
project_dir = path.mri(1:regexp(path.mri,'sub-\d\d\d[.]*')-1);
path.seg_mri = [project_dir,'derivatives',filesep,'sourcespace',filesep,sub,...
    filesep,sub,'_segmentedmri.mat'];
if ~exist(path.seg_mri,'file')
try
    mri = ft_read_mri([path.mri filesep file_name.mri '.mri']);
    mri = ft_convert_units(mri,'m');
catch
    try
        mri = ft_read_mri([path.mri filesep file_name.mri '.nii']);
        mri = ft_convert_units(mri,'m');
    catch
        error('MRI file is not in .mri or .nii format')
    end
end
    disp('Segmenting MRI...')
    cfg           = [];
    cfg.output    = {'brain','skull','scalp'};
    segmentedmri  = ft_volumesegment(cfg, mri);
    save(path.seg_mri,'segmentedmri')
    disp('Done!')
else
    disp('Loading Segmented MRI...')
    load(path.seg_mri)
    disp('Done!')
end

% seg_i = ft_datatype_segmentation(segmentedmri,'segmentationstyle','indexed');
% cfg = [];
% cfg.funparameter = 'seg';
% cfg.funcolormap  = gray(4); % distinct color per tissue
% cfg.location     = 'center';
% cfg.atlas        = seg_i;
% ft_sourceplot(cfg, seg_i);

% cfg = [];
% cfg.tissue = {'brain','skull','scalp'};
% cfg.numvertices = [3000 2000 1000];
% mesh_bem = ft_prepare_mesh(cfg,segmentedmri);
% figure 
% ft_plot_mesh(mesh_bem(1),'surfaceonly','yes','vertexcolor','none','facecolor',...
%            'skin','facealpha',0.5,'edgealpha',0.1)
% ft_plot_mesh(mesh_bem(2),'surfaceonly','yes','vertexcolor','none','facecolor',...
%            'skin','facealpha',0.5,'edgealpha',0.1)
% ft_plot_mesh(mesh_bem(3),'surfaceonly','yes','vertexcolor','none','facecolor',...
%            'skin','facealpha',0.5,'edgealpha',0.1)
% sphere_mesh.unit = 'mm';
% sphere_mesh.coordsys = 'ctf';  
% [v,f] = spheretri(10000);
% v = 70.7.*v + [15.5 -0.1 41.9];
% sphere_mesh.pos = v;
% sphere_mesh.tri = f;

cfg = [];
cfg.grad      = grad;
cfg.method    = 'singleshell';
cfg.tissue    = 'brain'; % will be constructed on the fly from white+grey+csf
vol = ft_prepare_headmodel(cfg, segmentedmri);

%% Visualise
headmodel = ft_convert_units(vol, 'm');
% cfg              = [];
% cfg.method = 'basedonpos';
% % cfg.resolution   = 1;
% cfg.grid.pos     = sourcepos.*1;
% cfg.grid.xgrid   = 'auto';
% cfg.grid.ygrid   = 'auto';
% cfg.grid.zgrid   = 'auto';
% cfg.tight        = 'yes';
% cfg.inwardshift  = -1.5;
% cfg.headmodel    = headmodel;
% template_grid    = ft_prepare_sourcemodel(cfg);

figure
ft_plot_sens(grad, 'style', '*b');
hold on
ft_plot_headmodel(headmodel, 'facecolor', 'cortex');
plot3(sourcepos(:,1),sourcepos(:,2),sourcepos(:,3),'rx')
%% Create leadfield
cfg                = [];
cfg.grad           = grad;
cfg.headmodel      = headmodel;
% cfg.resolution     = 0.4;
cfg.unit           = 'm';
cfg.grid.pos       = sourcepos.*1;
cfg.reducerank     = 2;
Lead_fields        = ft_prepare_leadfield(cfg);

inside_idx = find(Lead_fields.inside);
Leads = [];
for n = 1:length(inside_idx)
   Leads = [Leads, Lead_fields.leadfield{inside_idx(n)}]; 
end
L_reshaped = reshape(Leads,size(sensor_info.pos,1),3,[]);
% mesh_vn.vertices = headmodel.bnd.pos;
% mesh_vn.faces = headmodel.bnd.tri;
% vn = patchnormals(mesh_vn);
% 
% for n = 1:size(L_reshaped,3)
%     [~,nrst_mesh_pnt] = min(pdist2(sourcepos(n,:),headmodel.bnd.pos));
%     nrst_norm = vn(nrst_mesh_pnt,:);
%     x_cont = [1 0 0] - (dot([1 0 0],nrst_norm)).*nrst_norm;
%     y_cont = [0 1 0] - (dot([0 1 0],nrst_norm)).*nrst_norm;
%     z_cont = [0 0 1] - (dot([0 0 1],nrst_norm)).*nrst_norm;
%     
%     Lead_vector(:,n) = sum([L_reshaped(:,1,n).*norm(x_cont),L_reshaped(:,2,n).*norm(y_cont),...
%         L_reshaped(:,3,n).*norm(z_cont)],2).*1e-9;
% end
end

