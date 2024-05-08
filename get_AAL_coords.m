function [sourcepos_vox] = get_AAL_coords(AAL_regions,S)
%% [sourcepos_vox] = get_AAL_coords(AAL_regions,S)
%% get_AAL_coords generates voxel coordinates for each atlas region
% contained in AAL_regions ensuring they remain "inside" the set of valid
% source positions determined by fieldtrip.
%
% Requires:
%   AAL_regions - FieldTrip 4D volume containing each atlas mask (last
%                 volume contains the union of all atlas masks) read using
%                 ft_read_mri.
%             S - Struct containing fields
%                    <mri_file> (path to individual anatomy in the form 
%                             /project_dir/sub-XXX/ses-YYY/anat/...)
%                    <sensor_info> (struct with fields <pos> and <ors>
%                             containing sensor positions and orientations
% Expects 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AAL_regions = ft_convert_units(AAL_regions,'m');
all_regions = AAL_regions.anatomy(:,:,:,79)>0;

[all_regions_voxels(:,1),...
    all_regions_voxels(:,2),...
    all_regions_voxels(:,3)] = ind2sub(size(all_regions),find(all_regions));
all_regions_sourcepos = ft_warp_apply(AAL_regions.transform,all_regions_voxels);

%% MRI meshes
[path.mri, file_name.mri] = fileparts(S.mri_file);
sub_ = char(regexp(file_name.mri,'sub-\d\d\d','match')); %find sub-xyz
project_dir_ = path.mri(1:regexp(path.mri,'sub-\d\d\d[.]*')-1);
path.meshes = [project_dir_,'derivatives',filesep,'sourcespace',filesep,sub_,...
    filesep,sub_,'_meshes.mat'];

if ~exist(path.meshes,'file')
    try
        mri = ft_read_mri([path.mri filesep file_name.mri '.mri']);
        mri = ft_convert_units(mri,'m');
    catch
        try
            mri = ft_read_mri([path.mri filesep file_name.mri '.nii']);
            mri = ft_convert_units(mri,'m');
        catch
            error('Check MRI file is  in .mri or .nii format or that FieldTrip is added')
        end
    end
    
    path.segmentedmri = [project_dir_,'derivatives',filesep,'sourcespace',filesep,sub_,...
        filesep,sub_,'_segmentedmri.mat'];
    if ~exist(path.segmentedmri,'file')
        disp('Segmenting MRI...')
        cfg           = [];
        cfg.output    = {'brain','skull','scalp'};
        segmentedmri  = ft_volumesegment(cfg, mri);
        segmentedmri  = ft_convert_units(segmentedmri,'m');
        save(path.segmentedmri,'segmentedmri')
        disp('Done!')
    else
        disp('Loading Segmented MRI...')
        load(path.segmentedmri)
        disp('Done!')
    end
    cfg = [];
    cfg.tissue = {'brain','skull','scalp'};
    cfg.numvertices = [3000 2500 2500];
    mesh1 = ft_prepare_mesh(cfg,segmentedmri);
    for n = 1:size(mesh1,2)
        meshes(n).pnt = mesh1(n).pos;
        meshes(n).tri = mesh1(n).tri;
        meshes(n).unit = mesh1(n).unit;
        meshes(n).name = cfg.tissue{n};
        meshes(n) = ft_convert_units(meshes(n),'m');
    end
    save(path.meshes,'meshes','segmentedmri')
else
    load(path.meshes)
end

%%

grad.coilpos = S.sensor_info.pos.*1;
grad.coilori = S.sensor_info.ors;
grad.label = strsplit(num2str(1:size(S.sensor_info.pos)));
grad.chanpos = grad.coilpos;
grad.units = 'm';

cfg = [];
cfg.grad      = grad;
cfg.method    = 'singleshell';
cfg.tissue    = 'brain'; % will be constructed on the fly from white+grey+csf
vol = ft_prepare_headmodel(cfg, segmentedmri);

%% Visualise
headmodel = ft_convert_units(vol, 'm');

%% Create leadfield
cfg                = [];
cfg.grad           = grad;
cfg.headmodel      = headmodel;
% cfg.resolution     = 0.4;
cfg.unit           = 'm';
cfg.grid.pos       = all_regions_sourcepos.*1;
cfg.reducerank     = 2;
% Lead_fields        = ft_prepare_leadfield(cfg);
smodl = ft_prepare_sourcemodel(cfg);
%%
all_regions_inside = all_regions;
all_regions_inside(find(all_regions_inside)) = smodl.inside;
sourcepos_vox=nan(size(AAL_regions.anatomy,4)-1,3);
for reg_i = 1:size(AAL_regions.anatomy,4)-1
    AAL_regions.anatomy(:,:,:,reg_i) = AAL_regions.anatomy(:,:,:,reg_i).*all_regions_inside;
    region_voxels=[];
    [region_voxels(:,1),...
        region_voxels(:,2),...
        region_voxels(:,3)] = ind2sub(size(AAL_regions.anatomy(:,:,:,reg_i)),find(AAL_regions.anatomy(:,:,:,reg_i)));

    
    [~,medoid] = kmedoids(region_voxels,1);
    medioid_ind = find(sum((region_voxels == medoid),2)==3);
    if isempty(medioid_ind);error(sprintf('Medoid not in point set (region %d)',reg_i));end

%     figure(1);clf
%     scatter3(region_voxels(:,1),...
%         region_voxels(:,2),...
%         region_voxels(:,3),0.5,'g');
%     hold on
%     scatter3(region_voxels(medioid_ind,1),...
%         region_voxels(medioid_ind,2),...
%         region_voxels(medioid_ind,3),'r','MarkerFaceColor','r');
    sourcepos_vox(reg_i,:) = region_voxels(medioid_ind,:);
end