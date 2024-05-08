% save fieldtrip extracted brain

project_dir =  '.\Children';
% project_dir =  '.\Adults';

fieldtrip_path = [project_dir,filesep,'fieldtrip-20220906'];
if ~exist(fieldtrip_path,"dir")
    error("change 'fieldtrip_path' to valid path")
else
    addpath(fieldtrip_path)
end

ft_defaults;

for n=1:27

    sub = sprintf('1%2d',n);sub(sub == ' ') = '0';
    ses = '001';

    mriname = [project_dir,filesep,'Data',filesep,'BIDS',filesep,'sub-',sub,...
        filesep,'ses-',ses,filesep,'anat',filesep,'sub-',sub,'_anat.nii']
    mri = ft_read_mri(mriname);

    meshpath = [project_dir,filesep,'Data',filesep,'BIDS',filesep,'derivatives',...
        filesep,'sourcespace',filesep,'sub-',sub,filesep,'sub-',sub,'_meshes.mat'];

    load(meshpath,'segmentedmri')

    brainpath = [project_dir,filesep,'Data',filesep,'BIDS',filesep,'derivatives',...
        filesep,'sourcespace',filesep,'sub-',sub,filesep,'sub-',sub,'_brain.nii'];
    mri.anatomy = mri.anatomy .* segmentedmri.brain;

    ft_write_mri(brainpath,mri,'dataformat','nifti')
end

